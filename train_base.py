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
                        help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--collectionStrt', type=str, default='single', help='collection structure (single|multiple)')
    parser.add_argument('--collection', type=str,  help='dataset name')
    parser.add_argument('--trainCollection', type=str, help='train collection')
    parser.add_argument('--valCollection', type=str,  help='validation collection')
    parser.add_argument('--testCollection', type=str,  help='test collection')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1], help='overwrite existed file. (default: 0)')
    # model
    parser.add_argument('--model', type=str, default='nrccr', help='model name. (default: dual_encoding)')
    parser.add_argument('--concate', type=str, default='full', help='feature concatenation style. (full|reduced) full=level 1+2+3; reduced=level 2+3')
    parser.add_argument('--measure', type=str, default='cosine', help='measure method. (default: cosine)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (default: 0.2)')
    # text-side multi-level encoding
    parser.add_argument('--text_norm', action='store_true', help='normalize the text embeddings at last layer')
    # video-side multi-level encoding
    parser.add_argument('--visual_feature', type=str, default='resnet-152-img1k-flatten0_outputos', help='visual feature.')
    parser.add_argument('--visual_norm', action='store_true', help='normalize the visual embeddings at last layer')
    parser.add_argument('--gru_pool', type=str, default='mean', help='pooling on output of gru (mean|max)')
    # common space learning
    parser.add_argument('--text_mapping_layers', type=str, default='0-1536', help='text fully connected layers for common space learning. (default: 0-2048)')
    parser.add_argument('--visual_mapping_layers', type=str, default='0-1536', help='visual fully connected layers  for common space learning. (default: 0-2048)')
    # loss
    parser.add_argument('--loss_fun', type=str, default='mrl', help='loss function')
    parser.add_argument('--margin', type=float, default=0.2, help='rank loss margin')
    parser.add_argument('--direction', type=str, default='all', help='retrieval direction (all|t2v|v2t)')
    parser.add_argument('--max_violation', action='store_true', help='use max instead of sum in the rank loss')
    parser.add_argument('--cost_style', type=str, default='sum', help='cost style (sum, mean). (default: sum)')
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer. (default: rmsprop)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='learning rate decay rate. (default: 0.99)')
    parser.add_argument('--grad_clip', type=float, default=2, help='gradient clipping threshold')
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--val_metric', type=str, default='recall', help='performance metric for validation (mir|recall)')
    # misc
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of a training mini-batch.')
    parser.add_argument('--workers', type=int, default=5, help='Number of data loader workers.')
    parser.add_argument('--postfix', type=str, default='runs_0', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--log_step', type=int, default=10, help='Number of steps to print and record the log.')
    parser.add_argument('--cv_name', type=str, default='MM_2022', help='')
    #tag
    parser.add_argument('--framework', type=str, default='baseline')

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

    parser.add_argument('--tri_alpha', type=float, default=0.6,)
    parser.add_argument('--dtl_beta', type=float, default=0.4,)
    parser.add_argument('--l1_gama', type=float, default=1,)
    parser.add_argument('--back_w', type=float, default=1,)

    parser.add_argument('--model_type', type=str, default='video', help='video | img | ')

    parser.add_argument('--img_path', type=str)
    parser.add_argument('--img_encoder', type=str, default='clip', help='clip | resnet152 | ')
    parser.add_argument('--img_encoder_name', type=str)
    parser.add_argument('--img_encoder_input_dim', type=int,default=512)
    args = parser.parse_args()
    return args


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

