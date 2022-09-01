import os
import sys
import json
import torch
import logging

import validate
import tensorboard_logger as tb_logger
from model import get_model, get_we_parameter

import util.tag_data_provider_img as data

from basic.common import makedirsforfile, checkToSkip
from basic.util import read_dict, AverageMeter, LogCollector, log_config
from train_base import parse_args, train, save_checkpoint, decay_learning_rate, get_learning_rate, accuracy



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
                'val': '%s_%s_de2enc.caption.txt' % (opt.valCollection, opt.data_type.split('_')[0])}

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

    visual_encode_info = 'visual_feature_%s_visual_rnn_size_%d_visual_norm_%s' % \
            (opt.visual_feature, opt.visual_rnn_size, opt.visual_norm)
    visual_encode_info += "_kernel_sizes_%s_num_%s" % (opt.visual_kernel_sizes, opt.visual_kernel_num)

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


    opt.text_kernel_sizes = list(map(int, opt.text_kernel_sizes.split('-')))
    opt.visual_kernel_sizes = list(map(int, opt.visual_kernel_sizes.split('-')))
    opt.layer_list = list(opt.layer_list.split('-'))

    # caption
    caption_files = { x: os.path.join(rootpath, collections_pathname[x], 'TextData', cap_file[x])
                        for x in cap_file }
    caption_files_trans = {x: os.path.join(rootpath, collections_pathname[x], 'TextData', cap_file_trans[x])
                        for x in cap_file_trans }

    caption_files_back = {x: os.path.join(rootpath, collections_pathname[x], 'TextData', cap_file_back[x])
                           for x in cap_file_back}
    # Load visual features
    visual_feature_name = {'train': f'train-{opt.visual_feature}-avgpool.npy',
                           'val': f'val-{opt.visual_feature}-avgpool.npy'}
    visual_feat_path = {
        x: os.path.join(rootpath, collections_pathname[x], 'FeatureData', opt.visual_feature, visual_feature_name[x])
        for x in cap_file}

    print(visual_feat_path)
    exit()

    import numpy as np
    visual_feats = {x: np.load(visual_feat_path[x], encoding="latin1") for x in visual_feat_path}   #class 'numpy.ndarray'
    opt.visual_feat_dim = visual_feats['train'].shape[-1] # 2048

    # initialize word embedding
    opt.we_parameter = None

    # mapping layer structure
    opt.text_mapping_layers = list(map(int, opt.text_mapping_layers.split('-')))
    opt.visual_mapping_layers = list(map(int, opt.visual_mapping_layers.split('-')))
    if opt.concate == 'full':
        # opt.text_mapping_layers[0] = opt.bow_vocab_size + opt.text_rnn_size*2 + opt.text_kernel_num * len(opt.text_kernel_sizes)
        # opt.text_mapping_layers[0] = 768 + opt.text_rnn_size*2 + opt.text_kernel_num * len(opt.text_kernel_sizes)
        opt.text_mapping_layers[0] = opt.text_hidden_size
        opt.visual_mapping_layers[0] = opt.visual_feat_dim
    elif opt.concate == 'reduced':
        opt.text_mapping_layers[0] = opt.text_rnn_size*2 + opt.text_kernel_num * len(opt.text_kernel_sizes) 
        opt.visual_mapping_layers[0] = opt.visual_rnn_size*2 + opt.visual_kernel_num * len(opt.visual_kernel_sizes)
    else:
        raise NotImplementedError('Model %s not implemented' % opt.model)


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
    val_text_data_loader = data.get_txt_data_loader(opt, caption_files['val'], caption_files_trans['val'], opt.batch_size, opt.workers)

    # Construct the model
    model = get_model(opt.model)(opt)
    model.parallel()
    opt.we_parameter = None

    opt.resume = os.path.join(opt.resume, 'model_best.pth.tar')

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            logging.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            logging.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            # validate.validate(opt, tb_logger, data_loaders['val'], model, measure=opt.measure)
        else:
            logging.info("=> no checkpoint found at '{}'".format(opt.resume))


    # Train the Model
    best_rsum = 0
    no_impr_counter = 0
    lr_counter = 0
    best_epoch = None
    fout_val_metric_hist = open(os.path.join(opt.logger_name, 'val_metric_hist.txt'), 'w')
    for epoch in range(opt.num_epochs):
        logging.info('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, get_learning_rate(model.optimizer)[0]))
        logging.info('-'*10)
        # train for one epoch
        train(opt, data_loaders['train'], model, epoch)

        if opt.space == 'hybrid':
            rsum = validate.validate_hybrid(opt, tb_logger, val_vid_data_loader, val_text_data_loader, model, measure=opt.measure)
        elif opt.space == 'latent':
            rsum = validate.validate(opt, tb_logger, val_vid_data_loader, val_text_data_loader, model, measure=opt.measure)
        
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        logging.info(' * Current perf: {}'.format(rsum))
        logging.info(' * Best perf: {}'.format(best_rsum))
        logging.info('')
        fout_val_metric_hist.write('epoch_%d: %f\n' % (epoch, rsum))
        fout_val_metric_hist.flush()
        
        if is_best:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_epoch_%s.pth.tar'%epoch, prefix=opt.logger_name + '/', best_epoch=best_epoch)
            best_epoch = epoch

    
        lr_counter += 1
        decay_learning_rate(opt, model.optimizer, opt.lr_decay_rate)
        if not is_best:
            # When the validation performance decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            if lr_counter > 2:
                decay_learning_rate(opt, model.optimizer, 0.5)
                lr_counter = 0

        # Early stop occurs if the validation performance does not improve in ten consecutive epochs
        if not is_best:
            no_impr_counter += 1
        else:
            no_impr_counter = 0
        if no_impr_counter > 10:
            logging.info('Early stopping happended.\n')
            break

    fout_val_metric_hist.close()

    logging.info('best performance on validation: {}\n'.format(best_rsum))
    with open(os.path.join(opt.logger_name, 'val_metric.txt'), 'w') as fout:
        fout.write('best performance on validation: ' + str(best_rsum))

    # generate evaluation shell script
    if opt.testCollection == 'iacc.3':
        striptStr = ''.join(open('util/TEMPLATE_do_test_avs.sh').readlines())
        striptStr = striptStr.replace('@@@query_sets@@@', 'tv16.avs.txt,tv17.avs.txt,tv18.avs.txt')
    else:
        striptStr = ''.join(open('util/TEMPLATE_do_test_img.sh').readlines())
    striptStr = striptStr.replace('@@@rootpath@@@', rootpath)
    striptStr = striptStr.replace('@@@collectionStrt@@@', collectionStrt)
    striptStr = striptStr.replace('@@@testCollection@@@', collections_pathname['test'])
    striptStr = striptStr.replace('@@@logger_name@@@', opt.logger_name)
    striptStr = striptStr.replace('@@@overwrite@@@', str(opt.overwrite))

    # perform evaluation on test set
    runfile = 'do_test_%s_%s.sh' % (opt.model, collections_pathname['test'])
    open(runfile, 'w').write(striptStr + '\n')
    os.system('chmod +x %s' % runfile)
    os.system('./'+runfile)



if __name__ == '__main__':
    main()
