import json
import torch
import torch.utils.data as data
import numpy as np
import re

from basic.util import getVideoId

from transformers import BertTokenizer


VIDEO_MAX_LEN=64

def clean_str_cased(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()


def create_tokenizer():
    model_name_or_path = 'bert-base-multilingual-cased'
    do_lower_case = True
    cache_dir = 'data/cache_dir'
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir)
    return tokenizer

def tokenize_caption(tokenizer, raw_caption, cap_id, type='EN'):

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


def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    # videos, captions, cap_bows, idxs, cap_ids, video_ids, vid_tag = zip(*data)
    # BERT
    videos, bert_cap, bert_cap_trans, bert_cap_back, idxs, cap_ids, cap_ids_trans, cap_ids_back, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        videos_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0

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
    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
    # BERT
    text_data = (bert_target, lengths, words_mask), (bert_target_trans, lengths_trans, words_mask_trans)\
        , (bert_target_back, lengths_back, words_mask_back)

    return video_data, text_data, idxs, cap_ids, cap_ids_trans, cap_ids_back, video_ids

def collate_frame(data):

    videos, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)

    return video_data, idxs, video_ids


def collate_text(data, opt):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
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

    def __init__(self, opt, cap_file, cap_file_trans, cap_file_back, visual_feat, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = set()
        self.video2frames = video2frames
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                self.video_ids.add(video_id)
        self.visual_feat = visual_feat
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

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        str_ls = cap_id.split('#')
        if self.data_type == 'zh':
            tmp = '#zh#'
        elif self.data_type == 'google_enc2zh' or 'trans':
            tmp = '#enc2zh#'
        cap_id_trans = str_ls[0] + tmp + str_ls[2]
        tmp = '#enc2zh2enc#'
        cap_id_back = str_ls[0] + tmp + str_ls[2]
        video_id = getVideoId(cap_id)

        # video
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

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
        return frames_tensor, bert_tensor, bert_tensor_trans, bert_tensor_back, index, cap_id, cap_id_trans, cap_id_back, video_id

    def __len__(self):
        return self.length


class VisDataSet4DualEncoding(data.Dataset):
    """
    Load video frame features by pre-trained CNN model.
    """
    def __init__(self, visual_feat, video2frames=None, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        return frames_tensor, index, video_id

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
            tmp = '#zh#'
        elif 'zh2enc' in self.type:
            tmp = '#zh2enc#'
        elif 'enc2zh' in self.type:
            tmp = '#enc2zh#'
        else:
            tmp = '#zh#'

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


def get_train_data_loaders(opt, cap_files, cap_files_trans, cap_files_back, visual_feats, batch_size=100, num_workers=2, video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(opt, cap_files['train'], cap_files_trans['train'], cap_files_back['train'], visual_feats['train'], video2frames=video2frames['train'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files  if x=='train' }
    return data_loaders

def get_vis_data_loader(vis_feat, batch_size=100, num_workers=2, video2frames=None, video_ids=None):
    dset = VisDataSet4DualEncoding(vis_feat, video2frames, video_ids=video_ids)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_frame)
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
