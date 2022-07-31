import os
import logging

import evaluation
from model import get_model
from validate import norm_score, cal_perf

import util.tag_data_provider_img as data
import util.metrics as metrics

from test_base import test, process, parse_args



def main():
    opt = parse_args()

    rootpath, collectionStrt, collections_pathname, options, checkpoint, testCollection, output_dir, pred_error_matrix_file = process(opt=opt)

    # data loader prepare
    if collectionStrt == 'single':
        tmp = options.data_type.split('_')[-1].split('2')
        lang_type = tmp[-1] + '2' + tmp[0]
        # target sentence
        test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s%s_%s_2016.caption.txt' %(testCollection, opt.split, tmp[-1]))
        # source translation-sentence
        test_cap_trans = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s%s_google_%s_2016.caption.txt' %(testCollection, opt.split, lang_type))
    elif collectionStrt == 'multiple':
        test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s.caption.txt'%testCollection)
    else:
        raise NotImplementedError('collection structure %s not implemented' % collectionStrt)

    caption_files = {'test': test_cap}
    caption_files_trans = {'test': test_cap_trans}

    if options.img_encoder != 'clip':
        visual_feature_name = {'test': 'test_2016_flickr-resnet_152-avgpool.npy'}
        visual_feat_path = os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature, visual_feature_name['test'])

        import numpy as np
        visual_feats = {'test': np.load(visual_feat_path, encoding="latin1")}
        assert options.visual_feat_dim == visual_feats['test'].shape[-1]

    # Construct the model
    model = get_model(options.model)(options)
    model.parallel()
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()

    # set data loader
    image_id_name = {'test': 'test_id_2016.txt'}
    image_id_file = os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature, image_id_name['test'])
    test_image_ids_list = []
    with open(image_id_file) as f:
        for line in f.readlines():
            test_image_ids_list.append(line.strip())

    if options.img_encoder == 'clip':
        vid_data_loader = data.get_vis_data_loader(options, None, options.img_path, opt.batch_size, opt.workers,
                                                   image_ids=test_image_ids_list)
    else:
        vid_data_loader = data.get_vis_data_loader(options, visual_feats['test'], options.img_path, opt.batch_size, opt.workers, image_ids=test_image_ids_list)
    text_data_loader = data.get_txt_data_loader(options, caption_files['test'], caption_files_trans['test'], opt.batch_size, opt.workers, lang_type)

    test(options, model, vid_data_loader, text_data_loader, evaluation, metrics, logging, output_dir, norm_score,
         cal_perf, pred_error_matrix_file)


if __name__ == '__main__':
    main()
