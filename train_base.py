import os
import time
import torch
import shutil
import argparse

import tensorboard_logger as tb_logger

from basic.constant import ROOT_PATH
from basic.util import read_dict, AverageMeter, LogCollector, log_config
from basic.generic_utils import Progbar




def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('--collectionStrt', type=str, default='single', help='collection structure (single|multiple)')
    parser.add_argument('--collection', type=str, help='dataset name')
    parser.add_argument('--trainCollection', type=str, help='train collection')
    parser.add_argument('--valCollection', type=str, help='validation collection')
    parser.add_argument('--testCollection', type=str, help='test collection')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1], help='overwrite existed file. (default: 0)')
    # model
    parser.add_argument('--model', type=str, default='nrccr', help='model name. (default: dual_encoding)')
    parser.add_argument('--measure', type=str, default='cosine', help='measure method. (default: cosine)')
    parser.add_argument('--measure_2', type=str, default='jaccard', help='measure method. (default: cosine)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (default: 0.2)')
    parser.add_argument('--text_norm', action='store_true', help='normalize the text embeddings at last layer')
    parser.add_argument('--visual_feature', type=str, default='resnet-152-img1k-flatten0_outputos',
                        help='visual feature.')
    parser.add_argument('--visual_norm', action='store_true', help='normalize the visual embeddings at last layer')
    # common space learning
    parser.add_argument('--text_mapping_layers', type=str, default='0-1536',
                        help='text fully connected layers for common space learning. (default: 0-2048)')
    parser.add_argument('--visual_mapping_layers', type=str, default='0-1536',
                        help='visual fully connected layers  for common space learning. (default: 0-2048)')
    # loss
    parser.add_argument('--margin', type=float, default=0.2, help='rank loss margin')
    parser.add_argument('--direction', type=str, default='all', help='retrieval direction (all|t2v|v2t)')
    parser.add_argument('--max_violation', action='store_true', help='use max instead of sum in the rank loss')
    parser.add_argument('--cost_style', type=str, default='sum', help='cost style (sum, mean). (default: sum)')
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer. (default: rmsprop)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='learning rate decay rate. (default: 0.99)')
    parser.add_argument('--grad_clip', type=float, default=2, help='gradient clipping threshold')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--val_metric', type=str, default='recall',
                        help='performance metric for validation (mir|recall)')

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of a training mini-batch.')
    parser.add_argument('--workers', type=int, default=5, help='Number of data loader workers.')
    parser.add_argument('--postfix', type=str, default='runs_0', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--log_step', type=int, default=10, help='Number of steps to print and record the log.')

    # framework
    parser.add_argument('--framework', type=str, default='baseline')
    parser.add_argument('--cv_name', type=str, default='MM2022')
    parser.add_argument('--frozen', type=str, default='frozen')

    # transformer
    parser.add_argument('--text_num_attention', type=int, default=8)
    parser.add_argument('--text_hidden_size', type=int, default=1024)
    parser.add_argument('--video_num_attention', type=int, default=8)
    parser.add_argument('--video_hidden_size', type=int, default=1024)
    parser.add_argument('--video_pooling', type=str, default='mean')
    parser.add_argument('--text_pooling', type=str, default='mean')
    parser.add_argument('--text_layer', type=int, default=1)
    parser.add_argument('--video_layer', type=int, default=1)
    parser.add_argument('--layer_list', type=str, help='fine-tune_layer_list')
    parser.add_argument('--data_type', type=str, help='google_enc2zh')
    parser.add_argument('--sim_ratio', type=float, default=0.2, help='google_enc2zh')
    parser.add_argument('--scale', type=float, default=0.001, help='')
    parser.add_argument('--glr', type=float, default=0.001, help='lr of GAN')
    parser.add_argument('--disc_type', type=str, default='weak', help='weak | not so weak| strong')
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--optim', type=str, default='adam', help='adam | sgd | ')
    # loss weight
    parser.add_argument('--tri_alpha', type=float, default=0.6, )
    parser.add_argument('--dtl_beta', type=float, default=0.4, )
    parser.add_argument('--l1_gama', type=float, default=1, )
    parser.add_argument('--back_w', type=float, default=1, )

    # model_type
    parser.add_argument('--model_type', type=str, default='video', help='video | img | ')

    # about image
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--img_encoder', type=str, default='clip', help='clip | resnet152 | ')
    parser.add_argument('--img_encoder_name', type=str)
    parser.add_argument('--img_encoder_input_dim', type=int, default=512)

    args = parser.parse_args()
    return args


def preprocess():
    opt = parse_args()
    rootpath = opt.rootpath
    collectionStrt = opt.collectionStrt
    collection = opt.collection

    if collectionStrt == 'single':  # train,val data are in one directory
        opt.trainCollection = '%strain' % collection
        opt.valCollection = '%sval' % collection
        opt.testCollection = '%stest' % collection
        collections_pathname = {'train': collection, 'val': collection, 'test': collection}
    elif collectionStrt == 'multiple':  # train,val data are separated in multiple directories
        collections_pathname = {'train': opt.trainCollection, 'val': opt.valCollection, 'test': opt.testCollection}
    else:
        raise NotImplementedError('collection structure %s not implemented' % collectionStrt)

    return opt, rootpath, collectionStrt, collection, collections_pathname


def get_caption_file(rootpath, collections_pathname, cap_file, cap_file_trans, cap_file_back):

    caption_files = {x: os.path.join(rootpath, collections_pathname[x], 'TextData', cap_file[x])
                     for x in cap_file}
    caption_files_trans = {x: os.path.join(rootpath, collections_pathname[x], 'TextData', cap_file_trans[x])
                           for x in cap_file_trans}

    caption_files_back = {x: os.path.join(rootpath, collections_pathname[x], 'TextData', cap_file_back[x])
                          for x in cap_file_back}

    return caption_files, caption_files_trans, caption_files_back

def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    progbar = Progbar(len(train_loader.dataset))
    end = time.time()
    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger
        # Update the model
        b_size, loss, real_acc, fake_acc, real_loss, fake_loss, others_loss = model.train_emb(*train_data)
        progbar.add(b_size, values=[('loss', loss), ('en_acc', real_acc), ('zh_acc', fake_acc), ('en_loss', real_loss), ('zh_loss', fake_loss)])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', best_epoch=None):
    """save checkpoint at specific path"""
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    if best_epoch is not None:
        os.remove(prefix + 'checkpoint_epoch_%s.pth.tar'%best_epoch)


def decay_learning_rate(opt, optimizer, decay):
    """decay learning rate to the last LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay


def get_learning_rate(optimizer):
    """Return learning rate"""
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def process(opt, model, data_loaders, val_vid_data_loader, val_text_data_loader, validate, logging, collections_pathname, striptStr):

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
        logging.info('-' * 10)
        # train for one epoch
        train(opt, data_loaders['train'], model, epoch)

        rsum = validate.validate_hybrid(opt, tb_logger, val_vid_data_loader, val_text_data_loader, model,
                                        measure=opt.measure)

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
            }, is_best, filename='checkpoint_epoch_%s.pth.tar' % epoch, prefix=opt.logger_name + '/',
                best_epoch=best_epoch)
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

    # perform evaluation on test set
    runfile = 'do_test_%s_%s.sh' % (opt.model, collections_pathname['test'])
    open(runfile, 'w').write(striptStr + '\n')
    os.system('chmod +x %s' % runfile)
    os.system('./' + runfile)

