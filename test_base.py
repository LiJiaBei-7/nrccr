import torch
import json
import os
import logging
import sys
from basic.common import makedirsforfile, checkToSkip
from basic.util import read_dict, log_config
import argparse
from basic.constant import ROOT_PATH





def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--testCollection', type=str, help='test collection')
    parser.add_argument('--collectionStrt', type=str, default='single', help='collection structure (single|multiple)')
    parser.add_argument('--split', default='test', type=str, help='split, only for single-folder collection structure (val|test)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=100, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=64, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')

    args = parser.parse_args()
    return args


def process(opt, ):
    print(json.dumps(vars(opt), indent=2))

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
        output_dir = output_dir.replace('/%s/' % options.cv_name,
                                        '/results/%s/%s/' % (options.cv_name, trainCollection))
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    log_config(output_dir)
    logging.info(json.dumps(vars(opt), indent=2))

    return rootpath, collectionStrt, collections_pathname, options, checkpoint, testCollection,


def test(options, model, vid_data_loader, text_data_loader, evaluation, metrics, logging, output_dir, norm_score, cal_perf, pred_error_matrix_file):
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