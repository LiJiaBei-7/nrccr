import json
import torch
import torch.utils.data as data
import numpy as np

from basic.util import getVideoId
# from util.vocab import clean_str,clean_str_cased

from transformers import BertTokenizer


from PIL import Image
import clip


def read_one_image_feat(image_id, image_id_list, image_feat):
    index = image_id_list.index(image_id)
    return image_feat[index]

VIDEO_MAX_LEN=64


def create_tokenizer():
    model_name_or_path = 'bert-base-multilingual-cased'
    # model_name_or_path = 'bert-base-cased'
    # model_name_or_path = 'bert-large-uncased'
    do_lower_case = True
    cache_dir = 'data/cache_dir'
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir)
    return tokenizer

def tokenize_caption(tokenizer, raw_caption, cap_id, special_tokens=True, type='ZH'):
    # print(type, '--------')

    if(type == 'EN'):
        word_list = clean_str_cased(raw_caption)
        txt_caption = " ".join(word_list)
        # Remove whitespace at beginning and end of the sentence.
        txt_caption = txt_caption.strip()
        # Add period at the end of the sentence if not already there.
        try:
            if txt_caption[-1] not in [".", "?", "!"]:
                txt_caption += "."
        except:
            print(cap_id)
        txt_caption = txt_caption.capitalize()
        # tokens = tokenizer.tokenize(txt_caption)
        # if special_tokens:
        #     cls = [tokenizer.cls_token]
        #     sep = [tokenizer.sep_token]  # [SEP] token
        # tokens = cls + tokens + sep
        # # tokens = tokens[:self.max_text_words]
        # # Make sure that the last token is
        # # the [SEP] token
        # if special_tokens:
        #     tokens[-1] = tokenizer.sep_token
        #
        # ids = tokenizer.convert_tokens_to_ids(tokens)

        ids = tokenizer.encode(txt_caption, add_special_tokens=True)

    else:
        ids = tokenizer.encode(raw_caption, add_special_tokens=True)

    return ids



def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list


def collate_frame_gru_fn(data, img_type):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    # videos, captions, cap_bows, idxs, cap_ids, video_ids, vid_tag = zip(*data)
    # BERT
    # videos, bert_cap, bert_cap_trans, bert_cap_back, idxs, cap_ids, cap_ids_trans, cap_ids_back, video_ids = zip(*data)
    if img_type == 'clip':
        images_data, bert_cap, bert_cap_trans, bert_cap_back, idxs, cap_ids, cap_ids_trans, cap_ids_back, image_ids = zip(*data)
        images_ = torch.stack(images_data, dim=0)
    else:
        images_data, bert_cap, bert_cap_trans, bert_cap_back, idxs, cap_ids, cap_ids_trans, cap_ids_back, image_ids = zip(
            *data)
        dim = len(images_data[0])
        images_ = torch.zeros(len(images_data), dim)
        for i, image in enumerate(images_data):
            images_[i] = image


    # BERT
    if bert_cap[0] is not None:
        lengths = [len(cap) for cap in bert_cap]
        bert_target = torch.zeros(len(bert_cap), max(lengths)).long()
        words_mask = torch.zeros(len(bert_cap), max(lengths))
        for i, cap in enumerate(bert_cap):
            end = lengths[i]
            bert_target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        bert_target = None
        words_mask = None
        lengths = None

    # trans
    if bert_cap_trans[0] is not None:
        lengths_trans = [len(cap) for cap in bert_cap_trans]
        bert_target_trans = torch.zeros(len(bert_cap_trans), max(lengths_trans)).long()
        words_mask_trans = torch.zeros(len(bert_cap_trans), max(lengths_trans))
        for i, cap in enumerate(bert_cap_trans):
            end = lengths_trans[i]
            bert_target_trans[i, :end] = cap[:end]
            words_mask_trans[i, :end] = 1.0
    else:
        bert_target_trans = None
        words_mask_trans = None
        lengths_trans = None

    # back
    if bert_cap_back[0] is not None:
        lengths_back = [len(cap) for cap in bert_cap_back]
        bert_target_back = torch.zeros(len(bert_cap_back), max(lengths_back)).long()
        words_mask_back = torch.zeros(len(bert_cap_back), max(lengths_back))
        for i, cap in enumerate(bert_cap_back):
            end = lengths_back[i]
            bert_target_back[i, :end] = cap[:end]
            words_mask_back[i, :end] = 1.0
    else:
        bert_target_back = None
        words_mask_back = None
        lengths_back = None

    lengths = torch.IntTensor(lengths)
    lengths_trans = torch.IntTensor(lengths_trans)
    lengths_back = torch.IntTensor(lengths_back)
    vis_data = images_
    # BERT
    text_data = (bert_target, lengths, words_mask), (bert_target_trans, lengths_trans, words_mask_trans)\
        , (bert_target_back, lengths_back, words_mask_back)


    return vis_data, text_data, idxs, cap_ids, cap_ids_trans, cap_ids_back, image_ids

def collate_frame(data, img_type):

    images_data, idxs, image_ids = zip(*data)

    if img_type == 'clip':
        images_ = torch.stack(images_data, dim=0)
    else:
        dim = len(images_data[0])
        images_ = torch.zeros(len(images_data), dim)
        for i, image in enumerate(images_data):
            images_[i] = image

    return images_, idxs, image_ids


def collate_text(data, opt):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    # captions, cap_bows, idxs, cap_ids = zip(*data)
        bert_cap, bert_cap_trans, idxs, cap_ids = zip(*data)

    # BERT
    if bert_cap[0] is not None:
        lengths = [len(cap) for cap in bert_cap]
        bert_target = torch.zeros(len(bert_cap), max(lengths)).long()
        words_mask = torch.zeros(len(bert_cap), max(lengths))
        for i, cap in enumerate(bert_cap):
            end = lengths[i]
            bert_target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        bert_target = None
        words_mask = None
        lengths = None

    lengths = torch.IntTensor(lengths)
    # BERT
    text_data = (bert_target, lengths, words_mask)

    # trans
    if bert_cap_trans[0] is not None:
        lengths_trans = [len(cap) for cap in bert_cap_trans]
        bert_target_trans = torch.zeros(len(bert_cap_trans), max(lengths_trans)).long()
        words_mask_trans = torch.zeros(len(bert_cap_trans), max(lengths_trans))
        for i, cap in enumerate(bert_cap_trans):
            end = lengths_trans[i]
            bert_target_trans[i, :end] = cap[:end]
            words_mask_trans[i, :end] = 1.0
    else:
        bert_target_trans = None
        words_mask_trans = None
        lengths_trans = None
    lengths_trans = torch.IntTensor(lengths_trans)
    text_data = text_data, (bert_target_trans, lengths_trans, words_mask_trans)

    return text_data, idxs, cap_ids


class Dataset4DualEncoding(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, opt, cap_file, cap_file_trans, cap_file_back, visual_feat, image_path, image_id_file=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        # self.image_ids = set()
        self.image_ids = []
        self.img_path = image_path
        with open(image_id_file) as id_reader:
            for line in id_reader.readlines():
                img_id=line.strip('\n')
                self.image_ids.append(img_id)
                # self.img_path[img_id]=f'/home/zms/VisualSearch/multi30k/flickr30k-images/{img_id}.jpg'

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                # image_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                # self.image_ids.add(image_id)
        self.visual_feat = visual_feat # array
        self.length = len(self.cap_ids)

        # trans
        self.captions_trans = {}
        self.cap_ids_trans = []
        with open(cap_file_trans, 'r') as trans_reader:
            for line in trans_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions_trans[cap_id] = caption
                self.cap_ids_trans.append(cap_id)

        # back
        self.captions_back = {}
        self.cap_ids_back = []
        with open(cap_file_back, 'r') as back_reader:
            for line in back_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions_back[cap_id] = caption
                self.cap_ids_back.append(cap_id)

        # BERT
        self.tokenizer = create_tokenizer()

        self.data_type = opt.data_type
        self.img_type = opt.img_encoder

        _,self.preprocess = clip.load("ViT-B/32", device='cpu')

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        # yBsSqb4orpw_000000_000010#enc#5
        # Ptf_2VRj-V0_000122_000132#enc2zh#0
        str_ls = cap_id.split('#')
        if self.data_type == 'de':
            tmp = '#de#'
        elif self.data_type == 'google_enc2de':
            tmp = '#enc2de#'
            tmp_b = '#enc2de2enc#'
        elif self.data_type == 'google_enc2cs':
            tmp = '#enc2cs#'
            tmp_b = '#enc2cs2enc#'
        elif self.data_type == 'google_enc2fr':
            tmp = '#enc2fr#'
            tmp_b = '#enc2fr2enc#'

        cap_id_trans = str_ls[0] + tmp + str_ls[2]
        cap_id_back = str_ls[0] + tmp_b + str_ls[2]
        image_id = getVideoId(cap_id)

        if self.img_type == 'clip':
            image_tensor=Image.open(f'{self.img_path}/{image_id}.jpg')
            image_tensor = self.preprocess(image_tensor)
        else:
            image_vecs = read_one_image_feat(image_id, self.image_ids, self.visual_feat)
            image_tensor = torch.Tensor(image_vecs)

        # BERT
        caption = self.captions[cap_id]
        bert_ids = tokenize_caption(self.tokenizer, caption, cap_id, type='EN')
        bert_tensor = torch.Tensor(bert_ids)
        # trans
        caption_trans = self.captions_trans[cap_id_trans]
        bert_ids_trans = tokenize_caption(self.tokenizer, caption_trans, cap_id_trans, type='ZH')
        bert_tensor_trans = torch.Tensor(bert_ids_trans)
        # back
        caption_back = self.captions_back[cap_id_back]
        bert_ids_back = tokenize_caption(self.tokenizer, caption_back, cap_id_trans, type='ZH')
        bert_tensor_back = torch.Tensor(bert_ids_back)

        # BERT
        return image_tensor, bert_tensor, bert_tensor_trans, bert_tensor_back, index, cap_id, cap_id_trans, cap_id_back, image_id

    def __len__(self):
        return self.length


class VisDataSet4DualEncoding(data.Dataset):
    """
    Load video frame features by pre-trained CNN model.
    """
    def __init__(self, visual_feat, img_path, img_type, image_ids=None):
        self.visual_feat = visual_feat
        self.image_ids = image_ids
        self.img_path = img_path
        _, self.preprocess = clip.load("ViT-B/32", device='cpu')
        self.length = len(self.image_ids)
        self.type = img_type

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        if self.type == 'clip':
            image = self.preprocess(Image.open(f'{self.img_path}/{image_id}.jpg'))
        else:
            image_vecs = read_one_image_feat(image_id, self.image_ids, self.visual_feat)
            image = torch.Tensor(image_vecs)

        return image, index, image_id

    def __len__(self):
        return self.length

class TxtDataSet4DualEncoding(data.Dataset):
    """
    Load captions
    """
    def __init__(self, opt,  cap_file, cap_file_trans, lang_type):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.length = len(self.cap_ids)
        # BERT
        self.tokenizer = create_tokenizer()
        # trans
        self.captions_trans = {}
        self.cap_ids_trans = []
        with open(cap_file_trans, 'r') as trans_reader:
            for line in trans_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions_trans[cap_id] = caption
                self.cap_ids_trans.append(cap_id)

        self.type = lang_type

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        str_ls = cap_id.split('#')
        if self.type is None:
            tmp = '#de#'
        elif 'enc2de' in self.type:
            tmp = '#de#'
        elif 'enc2cs' in self.type:
            tmp = '#cs#'
        elif 'enc2fr' in self.type:
            tmp = '#fr#'
        elif 'de2enc' in self.type:
            tmp = '#de2enc#'
        elif 'cs2enc' in self.type:
            tmp = '#cs2enc#'
        elif 'fr2enc' in self.type:
            tmp = '#fr2enc#'
        else:
            tmp = '#de#'

        cap_id_trans = str_ls[0] + tmp + str_ls[2]

        # BERT
        caption = self.captions[cap_id]
        bert_ids = tokenize_caption(self.tokenizer, caption, cap_id)
        bert_tensor = torch.Tensor(bert_ids)
        # trans
        caption_trans = self.captions_trans[cap_id_trans]
        bert_ids_trans = tokenize_caption(self.tokenizer, caption_trans, cap_id_trans, type='ZH')
        bert_tensor_trans = torch.Tensor(bert_ids_trans)
        return bert_tensor, bert_tensor_trans,  index, cap_id

    def __len__(self):
        return self.length



def get_data_loaders(cap_files, visual_feats, tag_path, tag_vocab_path, vocab, bow2vec, batch_size=100, num_workers=2, video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], tag_path, tag_vocab_path, bow2vec, vocab, video2frames=video2frames['train']),
            'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], None, tag_vocab_path, bow2vec, vocab, video2frames=video2frames['val']) }

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files }
    return data_loaders


def get_train_data_loaders(opt, cap_files, cap_files_trans, cap_files_back, visual_feats, image_path, batch_size=100, num_workers=2, image_id_file=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(opt, cap_files['train'], cap_files_trans['train'], cap_files_back['train'], visual_feats['train'], image_path, image_id_file=image_id_file['train'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=lambda x: collate_frame_gru_fn(x, opt.img_encoder))
                        for x in cap_files  if x=='train' }
    return data_loaders



def get_vis_data_loader(opt, vis_feat, img_path, batch_size=100, num_workers=2, image_ids=None):
    dset = VisDataSet4DualEncoding(vis_feat, img_path, opt.img_encoder, image_ids=image_ids)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=lambda x: collate_frame(x, opt.img_encoder))
    return data_loader


def get_txt_data_loader(opt, cap_file, cap_file_trans, batch_size=50, num_workers=2, lang_type=None):
    dset = TxtDataSet4DualEncoding(opt, cap_file, cap_file_trans, lang_type)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=lambda x: collate_text(x, opt))
    return data_loader



if __name__ == '__main__':
    pass
