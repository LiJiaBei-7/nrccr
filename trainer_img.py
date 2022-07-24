import os
import sys
import json
import torch
import logging

import validate
import tensorboard_logger as tb_logger
from model import get_model

import util.tag_data_provider_img as data

from basic.common import makedirsforfile, checkToSkip
from basic.util import log_config
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


    cap_file = {'train': '%s_en.caption.txt' % opt.trainCollection,
                'val': '%s_google_%s2enc.caption.txt' % (opt.valCollection, opt.data_type.split('2')[-1])}

    cap_file_trans = {'train': '%s_%s.caption.txt' % (opt.trainCollection, opt.data_type),
                'val': '%s_%s.caption.txt' % (opt.valCollection, opt.data_type.split('2')[-1])}

    cap_file_back = {'train': '%s_%s2enc.caption.txt' % (opt.trainCollection, opt.data_type),
                      'val': '%s_de.caption.txt' % opt.valCollection}

    opt.collections_pathname = collections_pathname
    opt.cap_file = cap_file

    if opt.loss_fun == "mrl" and opt.measure == "cosine":
        assert opt.text_norm is True
        assert opt.visual_norm is True

    # checkpoint path
    opt.logger_name = os.path.join(rootpath, collections_pathname['train'], opt.cv_name, collections_pathname['val'], opt.framework, opt.postfix)
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

    # caption_files
    caption_files, caption_files_trans, caption_files_back = get_caption_file(rootpath, collections_pathname, cap_file, cap_file_trans, cap_file_back)

    # Load visual features
    visual_feature_name = {'train': f'train-{opt.visual_feature}-avgpool.npy',
                           'val': f'val-{opt.visual_feature}-avgpool.npy'}
    visual_feat_path = {
        x: os.path.join(rootpath, collections_pathname[x], 'FeatureData', opt.visual_feature, visual_feature_name[x])
        for x in cap_file}

    import numpy as np
    visual_feats = {x: np.load(visual_feat_path[x], encoding="latin1") for x in visual_feat_path}   #class 'numpy.ndarray'
    opt.visual_feat_dim = visual_feats['train'].shape[-1] # 2048

    # initialize word embedding
    opt.we_parameter = None

    # mapping layer structure
    opt.text_mapping_layers = list(map(int, opt.text_mapping_layers.split('-')))
    opt.visual_mapping_layers = list(map(int, opt.visual_mapping_layers.split('-')))

    opt.text_mapping_layers[0] = opt.text_hidden_size
    opt.visual_mapping_layers[0] = opt.visual_feat_dim


    # set data loader
    image_id_name = {'train': 'train_id.txt', 'val': 'val_id.txt'}
    image_id_file = {x: os.path.join(rootpath, collections_pathname[x], 'FeatureData', opt.visual_feature, image_id_name[x])
                     for x in cap_file}

    data_loaders = data.get_train_data_loaders(opt,
        caption_files, caption_files_trans, caption_files_back, visual_feats, opt.img_path, opt.batch_size, opt.workers, image_id_file=image_id_file)

    val_image_ids_list = []
    with open(image_id_file['val']) as f:
        for line in f.readlines():
            val_image_ids_list.append(line.strip())
    val_vid_data_loader = data.get_vis_data_loader(opt, visual_feats['val'], opt.img_path, opt.batch_size, opt.workers, image_ids=val_image_ids_list)
    val_text_data_loader = data.get_txt_data_loader(opt, caption_files['val'], caption_files_trans['val'], opt.batch_size, opt.workers, lang_type=opt.data_type.split('_'))

    # Construct the model
    model = get_model(opt.model)(opt)

    # generate evaluation shell script
    striptStr = ''.join(open('util/TEMPLATE_do_test_img.sh').readlines())
    striptStr = striptStr.replace('@@@rootpath@@@', rootpath)
    striptStr = striptStr.replace('@@@collectionStrt@@@', collectionStrt)
    striptStr = striptStr.replace('@@@testCollection@@@', collections_pathname['test'])
    striptStr = striptStr.replace('@@@logger_name@@@', opt.logger_name)
    striptStr = striptStr.replace('@@@overwrite@@@', str(opt.overwrite))

    process(opt, model, data_loaders, val_vid_data_loader, val_text_data_loader, validate, logging,
            collections_pathname, striptStr)

if __name__ == '__main__':
    main()
