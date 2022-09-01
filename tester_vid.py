import os
import sys
import json
import torch
import pickle
import logging

import evaluation
from model import get_model
from validate import norm_score, cal_perf

import util.tag_data_provider_vid as data
import util.metrics as metrics

from basic.util import read_dict, log_config
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from test_base import parse_args



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

    
    # collection setting
    testCollection = opt.testCollection
    collections_pathname = options.collections_pathname
    collections_pathname['test'] = testCollection

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    if 'checkpoints' in output_dir:
        output_dir = output_dir.replace('/checkpoints/', '/results/')
    else:
        output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/%s/' % (options.cv_name, trainCollection))
    result_pred_sents = os.path.join(output_dir, 'id.sent.score.txt')
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    log_config(output_dir)
    logging.info(json.dumps(vars(opt), indent=2))

    # data loader prepare
    test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s.caption.txt'%testCollection)
    if collectionStrt == 'single':
        lang_type = 'zh'
        test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s%s_google_zh2enc.caption.txt' %(testCollection, opt.split))
        test_cap_trans = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s%s_zh.caption.txt' %(testCollection, opt.split))
    elif collectionStrt == 'multiple':
        test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s.caption.txt'%testCollection)
    else:
        raise NotImplementedError('collection structure %s not implemented' % collectionStrt)

    caption_files = {'test': test_cap}
    caption_files_trans = {'test': test_cap_trans}
    img_feat_path = os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature)
    visual_feats = {'test': BigFile(img_feat_path)}
    assert options.visual_feat_dim == visual_feats['test'].ndims
    video2frames = {'test': read_dict(os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature, 'video2frames.txt'))}


    # Construct the model
    model = get_model(options.model)(options)
    model.parallel()
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()

    # set data loader
    video_ids_list = data.read_video_ids(caption_files['test'])
    vid_data_loader = data.get_vis_data_loader(visual_feats['test'], opt.batch_size, opt.workers, video2frames['test'], video_ids=video_ids_list)
    text_data_loader = data.get_txt_data_loader(options, caption_files['test'], caption_files_trans['test'], opt.batch_size, opt.workers, lang_type)

    # mapping

    video_embs, video_ids = evaluation.encode_text_or_vid(model.embed_vis, vid_data_loader)
    cap_embs, cap_trans_embs, caption_ids = evaluation.encode_text_hybrid(model.embed_txt, text_data_loader)

    v2t_gt, t2v_gt = metrics.get_gt(video_ids, caption_ids)

    logging.info("write into: %s" % output_dir)

    t2v_all_errors_1 = evaluation.cal_error(video_embs, cap_embs, options.measure)

    t2v_all_errors_2 = evaluation.cal_error(video_embs, cap_trans_embs, options.measure)


    for w in [1.0, 0.8, 0.5, 0.2, 0.0]:
        print(w, '------')
        t2v_all_errors_1 = norm_score(t2v_all_errors_1)
        t2v_all_errors_2 = norm_score(t2v_all_errors_2)
        t2v_tag_all_errors = w * t2v_all_errors_1 + (1 - w) * t2v_all_errors_2
        cal_perf(t2v_tag_all_errors, v2t_gt, t2v_gt)


if __name__ == '__main__':
    main()
