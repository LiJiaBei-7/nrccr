import os
import sys
import json
import torch
import pickle
import logging
import argparse

import evaluation
from model import get_model
from validate import norm_score, cal_perf

import util.tag_data_provider as data
import util.metrics as metrics

from basic.util import read_dict, log_config
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile

from test_base import test, process, parse_args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath, collectionStrt, collections_pathname, options, checkpoint, testCollection, output_dir, pred_error_matrix_file = process(opt=opt)

    # data loader prepare
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
    vid_feat_path = os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature)
    visual_feats = {'test': BigFile(vid_feat_path)}
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

    test(options, model, vid_data_loader, text_data_loader, evaluation, metrics, logging, output_dir, norm_score,
         cal_perf, pred_error_matrix_file)


if __name__ == '__main__':
    main()
