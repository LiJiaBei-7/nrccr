import os
import sys
import json
import torch
import logging
import argparse

import evaluation
from model import get_model
from validate import norm_score, cal_perf

import util.tag_data_provider_img as data
import util.metrics as metrics

from basic.util import read_dict, log_config
from basic.constant import ROOT_PATH
from basic.common import makedirsforfile, checkToSkip


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('--testCollection', type=str, help='test collection')
    parser.add_argument('--collectionStrt', type=str, default='single', help='collection structure (single|multiple)')
    parser.add_argument('--split', default='test', type=str,
                        help='split, only for single-folder collection structure (val|test)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1], help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=100, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=64, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str,
                        help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--space', default='latent', type=str)

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    # exit()

    rootpath = opt.rootpath
    collectionStrt = opt.collectionStrt
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']
    if opt.space == 'hybrid':
        options.space = 'hybrid'
        options.model = 'dual_encoding_hybrid'

    # collection setting
    testCollection = opt.testCollection
    collections_pathname = options.collections_pathname
    collections_pathname['test'] = testCollection

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    if 'checkpoints' in output_dir:
        output_dir = output_dir.replace('/checkpoints/', '/results/')
    else:
        output_dir = output_dir.replace('/%s/' % options.cv_name,
                                        '/results/%s/%s/' % (options.cv_name, trainCollection))
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    log_config(output_dir)
    logging.info(json.dumps(vars(opt), indent=2))

    # data loader prepare
    if collectionStrt == 'single':
        tmp = options.data_type.split('_')[-1].split('2')
        lang_type = tmp[-1] + '2' + tmp[0]
        test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData',
                                '%s%s_%s_2016.caption.txt' % (testCollection, opt.split, tmp[-1]))
        test_cap_trans = os.path.join(rootpath, collections_pathname['test'], 'TextData',
                                      '%s%s_google_%s_2016.caption.txt' % (testCollection, opt.split, lang_type))
    elif collectionStrt == 'multiple':
        test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s.caption.txt' % testCollection)
    else:
        raise NotImplementedError('collection structure %s not implemented' % collectionStrt)

    caption_files = {'test': test_cap}
    caption_files_trans = {'test': test_cap_trans}
    # img_feat_path = os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature)
    # visual_feats = {'test': BigFile(img_feat_path)}

    if options.img_encoder != 'clip':
        visual_feature_name = {'test': 'test_2016_flickr-resnet_152-avgpool.npy'}
        visual_feat_path = os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature,
                                        visual_feature_name['test'])

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
    image_id_file = os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature,
                                 image_id_name['test'])
    test_image_ids_list = []
    with open(image_id_file) as f:
        for line in f.readlines():
            test_image_ids_list.append(line.strip())

    if options.img_encoder == 'clip':
        vid_data_loader = data.get_vis_data_loader(options, None, options.img_path, opt.batch_size, opt.workers,
                                                   image_ids=test_image_ids_list)
    else:
        vid_data_loader = data.get_vis_data_loader(options, visual_feats['test'], options.img_path, opt.batch_size,
                                                   opt.workers, image_ids=test_image_ids_list)
    text_data_loader = data.get_txt_data_loader(options, caption_files['test'], caption_files_trans['test'],
                                                opt.batch_size, opt.workers, lang_type)

    # mapping
    video_embs, video_ids = evaluation.encode_text_or_vid(model.embed_vis, vid_data_loader)
    cap_embs, cap_trans_embs, caption_ids = evaluation.encode_text_hybrid(model.embed_txt, text_data_loader)

    v2t_gt, t2v_gt = metrics.get_gt(video_ids, caption_ids)

    logging.info("write into: %s" % output_dir)

    t2v_all_errors_1 = evaluation.cal_error(video_embs, cap_embs, options.measure)

    t2v_all_errors_2 = evaluation.cal_error(video_embs, cap_trans_embs, options.measure)

    for w in [1, 0.8, 0.5, 0.2, 0.0]:
        print(w, '------')
        t2v_all_errors_1 = norm_score(t2v_all_errors_1)
        t2v_all_errors_2 = norm_score(t2v_all_errors_2)
        t2v_tag_all_errors = w * t2v_all_errors_1 + (1 - w) * t2v_all_errors_2
        cal_perf(t2v_tag_all_errors, v2t_gt, t2v_gt)

    torch.save({'errors': t2v_tag_all_errors, 'videos': video_ids, 'captions': caption_ids}, pred_error_matrix_file)
    logging.info("write into: %s" % pred_error_matrix_file)


if __name__ == '__main__':
    main()
