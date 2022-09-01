import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from basic.bigfile import BigFile
from loss import TripletLoss, dtl_feat, DtlLoss
from transformers import BertModel
import copy
from transformer_cross import BertAttention, TrainablePositionalEncoding, LinearLayer



def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)


class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch normalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features

class video_transformer_encoding(nn.Module):
    def __init__(self, opt):
        super(video_transformer_encoding, self).__init__()
        self.input_drop = 0.1
        self.max_ctx_l = 80
        self.num_attention_heads = opt.video_num_attention
        self.hidden_size = opt.video_hidden_size

        self.input_proj_layer = LinearLayer(opt.visual_feat_dim, self.hidden_size, layer_norm=True,
                                            dropout=self.input_drop, relu=True)

        self.pos_embed_layer = TrainablePositionalEncoding(max_position_embeddings=self.max_ctx_l,
                                                         hidden_size=self.hidden_size, dropout=self.input_drop)

        self.encoder_layers = []
        self.layer = opt.video_layer
        self.encoder_layer = BertAttention(opt, self.num_attention_heads, self.hidden_size)
        for i in range(self.layer-1):
            self.encoder_layers.append(copy.deepcopy(self.encoder_layer).cuda())

        self.pooling = opt.video_pooling

    def forward(self, videos):
        videos, videos_origin, lengths, videos_mask = videos
        feat = self.input_proj_layer(videos)
        feat = self.pos_embed_layer(feat)
        mask = videos_mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        feat = self.encoder_layer(feat, feat, mask, mask).cuda()  # (N, L, D_hidden)
        for encoder_layer in self.encoder_layers:
            feat = encoder_layer(feat, feat, mask, mask)

        if self.pooling == 'mean':
            feat = F.avg_pool1d(feat.permute(0, 2, 1), feat.size(1)).squeeze(2)
        else:
            feat = F.max_pool1d(feat.permute(0, 2, 1), feat.size(1)).squeeze(2)
        return feat


class image_encoding(nn.Module):
    def __init__(self, opt):
        super(image_encoding, self).__init__()
        self.proj = nn.Linear(opt.visual_feat_dim, opt.visual_feat_dim)

    def forward(self, images):
        feat = F.relu(self.proj(images))
        return feat



import clip
from torchvision.models import resnet50
class image_encoding_clip(nn.Module):
    def __init__(self, opt):
        super(image_encoding_clip, self).__init__()
        self.img_encoder = opt.img_encoder
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
        self.encoder, _ = clip.load(opt.img_encoder_name, device='cuda')
        self.encoder = self.encoder.float()
        self.proj = nn.Linear(opt.img_encoder_input_dim, opt.visual_feat_dim)


    def forward(self, images):

        images = self.encoder.encode_image(images)
        images = torch.tensor(images,dtype=torch.float,requires_grad=True)

        return self.proj(images)



class Text_bert_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Text_bert_encoding, self).__init__()
        self.dropout = nn.Dropout(p=opt.dropout)
        self.txt_bert_params = {
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
        }

        txt_bert_config = 'bert-base-multilingual-cased'

        self.text_bert = BertModel.from_pretrained(txt_bert_config, return_dict=True, **self.txt_bert_params)
        # multi fc layers


    def forward(self, text, *args):
        # Embed word ids to vectors
        bert_caps, cap_mask = text

        batch_size, max_text_words = bert_caps.size()

        token_type_ids_list = []  # Modality id
        position_ids_list = []  # Position


        ids_size = (batch_size,)

        for pos_id in range(max_text_words):
            token_type_ids_list.append(torch.full(ids_size, 0, dtype=torch.long))
            position_ids_list.append(torch.full(ids_size, pos_id, dtype=torch.long))

        token_type_ids = torch.stack(token_type_ids_list, dim=1).cuda()
        position_ids = torch.stack(position_ids_list, dim=1).cuda()
        text_bert_output = self.text_bert(bert_caps,
                                        attention_mask=cap_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=None)

        # features = text_bert_output[0][:, 0]

        # mapping to common space
        del text
        torch.cuda.empty_cache()
        return text_bert_output



class Text_share(nn.Module):

    def __init__(self, opt):
        super(Text_share, self).__init__()
        self.max_ctx_l = 250
        self.bert_out_dim = 768
        self.input_drop = 0.1
        self.hidden_size = opt.text_hidden_size
        self.num_attention_heads = opt.text_num_attention
        self.input_proj_layer = LinearLayer(self.bert_out_dim, self.hidden_size, layer_norm=True,
                                            dropout=self.input_drop, relu=True)

        self.pos_embed_layer = TrainablePositionalEncoding(max_position_embeddings=self.max_ctx_l,
                                                           hidden_size=self.hidden_size, dropout=self.input_drop)

        self.layer = opt.text_layer
        self.encoder_layer = BertAttention(opt, self.num_attention_heads, self.hidden_size)

        self.text_bert = Text_bert_encoding(opt)

        self.pooling = opt.text_pooling

    def forward(self, texts, reference=False):

        if reference:
            text, text_trans = texts
        else:
            text, text_trans, text_back = texts


        bert_caps, lengths, cap_mask = text
        bert_caps_trans, lengths_trans, cap_mask_trans = text_trans

        # EN
        bert_out = self.text_bert((bert_caps, cap_mask))
        bert_seq = bert_out.last_hidden_state
        feat = self.input_proj_layer(bert_seq)
        feat = self.pos_embed_layer(feat)
        mask = cap_mask.unsqueeze(1)

        # trans
        bert_out_trans = self.text_bert((bert_caps_trans, cap_mask_trans))
        bert_seq_trans = bert_out_trans.last_hidden_state
        feat_trans = self.input_proj_layer(bert_seq_trans)
        feat_trans = self.pos_embed_layer(feat_trans)
        mask_trans = cap_mask_trans.unsqueeze(1)

        feat_self = self.encoder_layer(feat, feat, mask, mask).cuda()  # (N, L, D_hidden)
        feat_self_trans = self.encoder_layer(feat_trans, feat_trans, mask_trans, mask_trans).cuda()  # (N, L, D_hidden)


        if self.pooling == 'mean':
            feat_vec = F.avg_pool1d(feat_self.permute(0, 2, 1), feat_self.size(1)).squeeze(2)
            feat_trans_vec = F.avg_pool1d(feat_self_trans.permute(0, 2, 1), feat_self_trans.size(1)).squeeze(2)

        if reference:
            return feat_vec, feat_trans_vec

        feat_cross = self.encoder_layer(feat, feat_trans, mask, mask_trans, cross=True)
        del feat, feat_self, feat_self_trans

        # back
        bert_caps_back, lengths_back, cap_mask_back = text_back
        bert_out_back = self.text_bert((bert_caps_back, cap_mask_back))
        bert_seq_back = bert_out_back.last_hidden_state
        feat_back = self.input_proj_layer(bert_seq_back)
        feat_back = self.pos_embed_layer(feat_back)
        mask_back = cap_mask_back.unsqueeze(1)

        feat_self_back = self.encoder_layer(feat_back, feat_back, mask_back, mask_back)

        if self.pooling == 'mean':
            # pooling_cat
            feat_cross = torch.cat((feat_cross, feat_trans), 1)
            feat_cross_vec = F.avg_pool1d(feat_cross.permute(0, 2, 1), feat_cross.size(1)).squeeze(2)
            feat_back_vec = F.avg_pool1d(feat_self_back.permute(0, 2, 1), feat_self_back.size(1)).squeeze(2)

        return (feat_vec, feat_trans_vec, feat_cross_vec, feat_back_vec), (bert_seq, bert_seq_trans)




class Latent_mapping(nn.Module):
    """
    Latent space mapping (Conference version)
    """
    def __init__(self, mapping_layers, dropout, l2norm=True):
        super(Latent_mapping, self).__init__()
        
        self.l2norm = l2norm
        # visual mapping
        self.mapping = MFC(mapping_layers, dropout, have_bn=True, have_last_bn=True)


    def forward(self, features):

        # mapping to latent space
        latent_features = self.mapping(features)
        if self.l2norm:
            latent_features = l2norm(latent_features)

        return latent_features




class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoding.state_dict(), self.text_encoding.state_dict(), self.vid_mapping.state_dict(), self.text_mapping.state_dict(), self.AdvAgent.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])
        self.vid_mapping.load_state_dict(state_dict[2])
        self.text_mapping.load_state_dict(state_dict[3])
        self.AdvAgent.load_state_dict(state_dict[4])

    def train_start(self):
        """switch to train mode
        """
        self.vid_encoding.train()
        self.text_encoding.train()
        self.vid_mapping.train()
        self.text_mapping.train()
        self.AdvAgent.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.vid_encoding.eval()
        self.text_encoding.eval()
        self.vid_mapping.eval()
        self.text_mapping.eval()
        self.AdvAgent.eval()


    def init_info(self, opt):

        # init gpu
        if torch.cuda.is_available():
            self.vid_encoding.cuda()
            self.text_encoding.cuda()
            self.vid_mapping.cuda()
            self.text_mapping.cuda()
            self.AdvAgent.cuda()
            cudnn.benchmark = True


        if opt.frozen == 'frozen':
            print(opt.frozen, '--------')
            text_param = []
            bert_name = 'text_bert.text_bert'
            # finetune_layer
            layer_list = opt.layer_list
            print(layer_list)
            for name, param in self.text_encoding.named_parameters():
                if bert_name in name and not any(layer in name for layer in layer_list):
                    param.requires_grad = False
                else:
                    text_param.append(param)

        elif opt.frozen == 'all_frozen':
            print(opt.frozen, '--------')
            text_param = []
            bert_name = 'text_bert.text_bert'
            for name, param in self.text_encoding.named_parameters():
                if bert_name in name:
                    param.requires_grad = False
                else:
                    text_param.append(param)

        else:
            print(opt.frozen, '--------')
            text_param = list(self.text_encoding.parameters())

        # init params
        params = list(self.vid_encoding.parameters())
        params += text_param
        params += list(self.vid_mapping.parameters())
        params += list(self.text_mapping.parameters())
        self.params = params

        # print structure
        print(self.vid_encoding)
        print(self.text_encoding)
        print(self.vid_mapping)
        print(self.text_mapping)
        print(self.AdvAgent)


class Model(BaseModel):
    """
    dual encoding network
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.model_type = opt.model_type

        if self.model_type == 'img':
            if opt.img_encoder == 'clip':
                self.vid_encoding = image_encoding_clip(opt)
            else:
                self.vid_encoding = image_encoding(opt)
        else:
            self.vid_encoding = video_transformer_encoding(opt)
        self.text_encoding = Text_share(opt)

        # lang-agnostic learning
        from adv import Adversarial
        kwargs = {'opt': opt, 'input_size': opt.text_hidden_size,
                  'train_level': 'sent', 'train_type': 'GAN',
                  'reverse_grad': False, 'nclass': 2, 'scale': opt.scale,
                  'optim': 'adam', 'lr': opt.glr, 'betas': (0.9, 0.999), 'gamma': 0, 'eps': 1e-8,
                  'momentum': opt.momentum, 'disc_type': opt.disc_type}
        self.AdvAgent = Adversarial(**kwargs)

        if torch.cuda.is_available():
            print('use', torch.cuda.device_count(), 'gpus')
            self.vid_encoding.cuda()
            self.text_encoding.cuda()
            cudnn.benchmark = True

        self.vid_mapping = Latent_mapping(opt.visual_mapping_layers, opt.dropout)
        self.text_mapping = Latent_mapping(opt.text_mapping_layers, opt.dropout)

        self.init_info(opt)

        # Loss and Optimizer
        if opt.loss_fun == 'mrl':
            self.criterion = TripletLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation,
                                         cost_style=opt.cost_style,
                                         direction=opt.direction)

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params, lr=opt.learning_rate)

        self.dtl_feat = dtl_feat()
        self.dtl_criterion = DtlLoss()

        self.tri_alpha = opt.tri_alpha
        self.dtl_beta = opt.dtl_beta
        self.l1_gama = opt.l1_gama
        self.back_w = opt.back_w

        self.Eiters = 0

    def parallel(self):
        self.vid_encoding = nn.parallel.DataParallel(self.vid_encoding)
        self.text_encoding = nn.parallel.DataParallel(self.text_encoding)

    def forward_loss(self, cap_embs, cap_bert_embs, vid_emb, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        cap_emb, cap_emb_trans, cap_emb_cross, cap_emb_back = cap_embs
        cap_bert_emb, cap_bert_emb_trans = cap_bert_embs
        loss_tri = self.criterion(cap_emb, vid_emb)
        loss_tri_trans = self.criterion(cap_emb_trans, vid_emb) * self.tri_alpha

        loss_dtl = self.dtl_criterion(cap_emb_cross.detach(), cap_emb_trans, vid_emb) * self.dtl_beta
        loss_feat = self.dtl_feat(cap_emb_cross, cap_emb_trans) * self.l1_gama
        loss_contrastive = self.criterion(cap_emb, cap_emb_back) * self.back_w

        real_idx = 1
        fake_idx = 0
        # update discriminator
        real_loss, fake_loss, real_acc, fake_acc = self.AdvAgent.update(cap_bert_emb.detach(),
                                                                   cap_bert_emb_trans.detach(),
                                                          real_idx, fake_idx)
        # update encoder
        others_loss = self.AdvAgent.gen_loss(cap_bert_emb, cap_bert_emb_trans, real_idx, fake_idx)

        loss = loss_tri + loss_tri_trans + loss_dtl + loss_contrastive + others_loss + loss_feat

        self.logger.update('Le', loss.item(), vid_emb.size(0))
        self.logger.update('Le_tri', loss_tri.item(), vid_emb.size(0))
        self.logger.update('Le_tri_trans', loss_tri_trans.item(), vid_emb.size(0))
        self.logger.update('loss_dtl', loss_dtl.item(), vid_emb.size(0))
        self.logger.update('loss_feat', loss_feat.item(), vid_emb.size(0))
        self.logger.update('real_loss', real_loss, vid_emb.size(0))
        self.logger.update('fake_loss', fake_loss, vid_emb.size(0))
        self.logger.update('others_loss', others_loss, vid_emb.size(0))
        # self.logger.update('loss_mul', loss_mul.item(), vid_emb.size(0))
        self.logger.update('loss_contrastive', loss_contrastive.item(), vid_emb.size(0))
        self.logger.update('real_acc', real_acc, vid_emb.size(0))
        self.logger.update('fake_acc', fake_acc, vid_emb.size(0))

        return loss, real_acc, fake_acc, real_loss, fake_loss, others_loss


    def forward_emb(self, visual, targets, volatile=False, *args):

        """Compute the video and caption embeddings
        """
        # -------video
        if self.model_type == 'video':
            # video data
            frames, mean_origin, video_lengths, vidoes_mask = visual
            frames = Variable(frames, volatile=volatile)
            if torch.cuda.is_available():
                frames = frames.cuda()

            mean_origin = Variable(mean_origin, volatile=volatile)
            if torch.cuda.is_available():
                mean_origin = mean_origin.cuda()

            vidoes_mask = Variable(vidoes_mask, volatile=volatile)
            if torch.cuda.is_available():
                vidoes_mask = vidoes_mask.cuda()
            visual_data = (frames, mean_origin, video_lengths, vidoes_mask)

        # -------image
        elif self.model_type == 'img':
            # video data
            images = Variable(visual, volatile=volatile)
            if torch.cuda.is_available():
                visual_data = images.cuda()

        target, target_trans, target_back = targets
        bert_caps, lengths, cap_masks = target
        if bert_caps is not None:
            bert_caps = Variable(bert_caps, volatile=volatile)
            if torch.cuda.is_available():
                bert_caps = bert_caps.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        text_data = (bert_caps, lengths, cap_masks)

        # -------trans
        bert_caps_trans, lengths_trans, cap_masks_trans = target_trans
        if bert_caps_trans is not None:
            bert_caps_trans = Variable(bert_caps_trans, volatile=volatile)
            if torch.cuda.is_available():
                bert_caps_trans = bert_caps_trans.cuda()

        if cap_masks_trans is not None:
            cap_masks_trans = Variable(cap_masks_trans, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks_trans = cap_masks_trans.cuda()
        text_data_trans = (bert_caps_trans, lengths_trans, cap_masks_trans)

        # -------back
        bert_caps_back, lengths_back, cap_masks_back = target_back
        if bert_caps_back is not None:
            bert_caps_back = Variable(bert_caps_back, volatile=volatile)
            if torch.cuda.is_available():
                bert_caps_back = bert_caps_back.cuda()

        if cap_masks_back is not None:
            cap_masks_back = Variable(cap_masks_back, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks_back = cap_masks_back.cuda()
        text_data_back = (bert_caps_back, lengths_back, cap_masks_back)
        text_data = (text_data, text_data_trans, text_data_back)

        vid_emb = self.vid_mapping(self.vid_encoding(visual_data))
        txt_embs = self.text_encoding(text_data)
        txt_embs, txt_bert_embs = txt_embs

        if len(txt_embs) > 1:
            cap_embs = ()
            for k in txt_embs:
                cap_embs = cap_embs + (self.text_mapping(k),)
        else:
            cap_embs = self.text_mapping(txt_embs)

        return vid_emb, cap_embs, txt_bert_embs

    def embed_vis(self, vis_data, volatile=True):
        """Compute the video embeddings
        """
        # video data
        if self.model_type == 'video':
            frames, mean_origin, video_lengths, vidoes_mask = vis_data
            frames = Variable(frames, volatile=volatile)
            if torch.cuda.is_available():
                frames = frames.cuda()

            mean_origin = Variable(mean_origin, volatile=volatile)
            if torch.cuda.is_available():
                mean_origin = mean_origin.cuda()

            vidoes_mask = Variable(vidoes_mask, volatile=volatile)
            if torch.cuda.is_available():
                vidoes_mask = vidoes_mask.cuda()
            vis_data = (frames, mean_origin, video_lengths, vidoes_mask)


        return self.vid_mapping(self.vid_encoding(vis_data))

    def embed_txt(self, txt_data, volatile=True):
        """Compute the caption embeddings
        """
        txt_data, target_trans = txt_data
        bert_caps_trans, lengths_trans, cap_masks_trans = target_trans
        if bert_caps_trans is not None:
            bert_caps_trans = Variable(bert_caps_trans, volatile=volatile)
            if torch.cuda.is_available():
                bert_caps_trans = bert_caps_trans.cuda()

        if cap_masks_trans is not None:
            cap_masks_trans = Variable(cap_masks_trans, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks_trans = cap_masks_trans.cuda()
        txt_data_trans = (bert_caps_trans, lengths_trans, cap_masks_trans)

        # BERT
        bert_caps, lengths, cap_masks = txt_data
        if bert_caps is not None:
            bert_caps = Variable(bert_caps, volatile=volatile)
            if torch.cuda.is_available():
                bert_caps = bert_caps.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        txt_data = (bert_caps, lengths, cap_masks)
        txt_data = (txt_data, txt_data_trans)

        txt_embs = self.text_encoding(txt_data, reference=True)
        cap_embs = ()
        for k in txt_embs:
            cap_embs = cap_embs + (self.text_mapping(k),)

        return cap_embs


    def train_emb(self, videos, captions, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb, cap_emb, cap_bert_emb = self.forward_emb(videos, captions, False)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss, real_acc, feak_acc, real_loss, fake_loss, others_loss = self.forward_loss(cap_emb, cap_bert_emb, vid_emb)
        loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        return vid_emb.size(0), loss_value, real_acc, feak_acc, real_loss, fake_loss, others_loss


NAME_TO_MODELS = {'nrccr': Model}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]
