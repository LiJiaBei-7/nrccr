import os
import sys
import json
import torch
import logging

import validate
import tensorboard_logger as tb_logger
from model import get_model

import util.tag_data_provider as data

from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from basic.util import read_dict, log_config

from train_base import parse_args, get_caption_file, process



def main():
    opt = parse_args()

    rootpath = opt.rootpath
    collectionStrt = opt.collectionStrt
    collection = opt.collection

    if collectionStrt == 'single': # train,val data are in one directory
        opt.trainCollection = '%strain' % collection
        opt.valCollection = '%sval' % collection
        opt.testCollection = '%stest' % collection
        collections_pathname = {'train': collection, 'val': collection, 'test': collection}
    elif collectionStrt == 'multiple': # train,val data are separated in multiple directories
        collections_pathname = {'train': opt.trainCollection, 'val': opt.valCollection, 'test': opt.testCollection}
    else:
        raise NotImplementedError('collection structure %s not implemented' % collectionStrt)


    cap_file = {'train': '%s.caption.txt' % opt.trainCollection,
                'val': '%s_%s_zh2enc.caption.txt' % (opt.valCollection, opt.data_type.split('_')[0])}

    cap_file_trans = {'train': '%s_%s.caption.txt' % (opt.trainCollection, opt.data_type),
                'val': '%s_zh.caption.txt' % opt.valCollection}

    cap_file_back = {'train': '%s_google_enc2zh2enc.caption.txt' % (opt.trainCollection),
                      'val': '%s_zh.caption.txt' % opt.valCollection}
                
    opt.collections_pathname = collections_pathname
    opt.cap_file = cap_file

    if opt.loss_fun == "mrl" and opt.measure == "cosine":
        assert opt.text_norm is True
        assert opt.visual_norm is True

    framework = opt.framework

    opt.logger_name = os.path.join(rootpath, collections_pathname['train'], opt.cv_name, collections_pathname['val'], framework, opt.postfix)
    logging.info(opt.logger_name)

    if checkToSkip(os.path.join(opt.logger_name, 'model_best.pth.tar'), opt.overwrite):
        sys.exit(0)
    if checkToSkip(os.path.join(opt.logger_name, 'val_metric.txt'), opt.overwrite):
        sys.exit(0)
    makedirsforfile(os.path.join(opt.logger_name, 'val_metric.txt'))

    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    log_config(opt.logger_name)
    tb_logger.configure(opt.logger_name, flush_secs=5)
    logging.info(json.dumps(vars(opt), indent=2))


    opt.layer_list = list(opt.layer_list.split('-'))

    # caption
    caption_files, caption_files_trans, caption_files_back = get_caption_file(rootpath, collections_pathname, cap_file, cap_file_trans, cap_file_back)

    # Load visual features
    visual_feat_path = {x: os.path.join(rootpath, collections_pathname[x], 'FeatureData', opt.visual_feature)
                        for x in cap_file }

    visual_feats = {x: BigFile(visual_feat_path[x]) for x in visual_feat_path}
    opt.visual_feat_dim = visual_feats['train'].ndims

    # initialize word embedding
    opt.we_parameter = None

    # mapping layer structure
    opt.text_mapping_layers = list(map(int, opt.text_mapping_layers.split('-')))
    opt.visual_mapping_layers = list(map(int, opt.visual_mapping_layers.split('-')))
    opt.text_mapping_layers[0] = opt.text_hidden_size
    opt.visual_mapping_layers[0] = opt.video_hidden_size



    # set data loader
    video2frames = {x: read_dict(os.path.join(rootpath, collections_pathname[x], 'FeatureData', opt.visual_feature, 'video2frames.txt'))
                    for x in cap_file}
    data_loaders = data.get_train_data_loaders(opt,
        caption_files, caption_files_trans, caption_files_back, visual_feats, opt.batch_size, opt.workers, video2frames=video2frames)
    val_video_ids_list = data.read_video_ids(caption_files['val'])
    val_vid_data_loader = data.get_vis_data_loader(visual_feats['val'], opt.batch_size, opt.workers, video2frames['val'], video_ids=val_video_ids_list)
    val_text_data_loader = data.get_txt_data_loader(opt, caption_files['val'], caption_files_trans['val'], opt.batch_size, opt.workers)

    # Construct the model
    model = get_model(opt.model)(opt)

    striptStr = ''.join(open('util/TEMPLATE_do_test.sh').readlines())
    striptStr = striptStr.replace('@@@rootpath@@@', rootpath)
    striptStr = striptStr.replace('@@@collectionStrt@@@', collectionStrt)
    striptStr = striptStr.replace('@@@testCollection@@@', collections_pathname['test'])
    striptStr = striptStr.replace('@@@logger_name@@@', opt.logger_name)
    striptStr = striptStr.replace('@@@overwrite@@@', str(opt.overwrite))

    process(opt, model, data_loaders, val_vid_data_loader, val_text_data_loader, validate, logging,
            collections_pathname, striptStr)


if __name__ == '__main__':
    main()
