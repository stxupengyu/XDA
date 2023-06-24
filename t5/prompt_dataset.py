from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from pathlib import Path
import re
import torch
import numpy as np
import random
import copy
from tqdm import tqdm
import nltk

def nltk_line_tokenizer(line):
    return nltk.word_tokenize(line)

class NLGMixSenClsDataset(Dataset):

    SKIP_ATTRIBUTES = ['gt_x', 'gt_y']

    def __init__(self, _C, _A, train_texts, tokenizer):
        self.texts = train_texts
        self.tokenizer = tokenizer
        self.config = _C
        self.mask_ratio = _A.mask_ratio
        print("Data Size %d" % len(self.texts))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_i = self.texts[idx]
        x_np, y_np, input_y, input_x = self.gen_masked_pair(text_i)
        task_index = 0
        return x_np, y_np, input_y, input_x, task_index

    def gen_masked_pair(self, text_i):
        input_x, _, input_y = mask_text(text_i, self.mask_ratio)
        y_np = self.tokenizer(input_y, return_tensors="np")['input_ids'][0, :self.config.max_length]
        x_np = self.tokenizer(input_x, return_tensors="np")['input_ids'][0, :self.config.max_length]
        return x_np, y_np, input_y, input_x

def mask_text(text, mask_ratio=0.5, cnt=0,
              substitute_verbalizers=['<extra_id_{}>'.format(i) for i in range(300)],
              allow_substitute_punctuation=False, at_least_one=False, unchanged_phrases=[], changed_word_list=[]):
    '''
    input: sentence without mask
    output:
    movie and art I really enjoy <extra_id_0> this <extra_id_1>
    <extra_id_0> this movie and art I really enjoy<extra_id_1> movie and art
    '''

    tokens = nltk_line_tokenizer(text)
    # print('text', text)
    # print('tokens', tokens)
    # assert 0
    n = len(tokens)
    unchanged_phrases = [x.lower() for x in unchanged_phrases]
    splited_unchanged_phrases = [nltk_line_tokenizer(x.lower()) for x in unchanged_phrases]
    changed_word_list = [x.lower() for x in changed_word_list]
    if allow_substitute_punctuation:
        candidate_idxs = np.ones(n)
        for i in range(n):
            for splited_unchanged_phrase in splited_unchanged_phrases:
                if ' '.join(tokens[i:i + len(splited_unchanged_phrase)]).lower() == ' '.join(
                        splited_unchanged_phrase):
                    candidate_idxs[i:i + len(splited_unchanged_phrase)] = 0
        candidate_idxs = [i for (i, x) in enumerate(candidate_idxs) if x == 1]
        # candidate_idxs=[i for i in range(n) if tokens[i].lower() not in unchanged_word_list]
        idxs_should_be_changed = [i for i in range(n) if tokens[i].lower() in changed_word_list]
        n = len(candidate_idxs)
        indices = sorted(list(set(random.sample(candidate_idxs, int(n * mask_ratio)) + idxs_should_be_changed)))
        # indices=sorted(random.sample(range(n),int(n*mask_ratio)))
    else:
        candidate_idxs = np.ones(n)
        for i in range(n):
            for splited_unchanged_phrase in splited_unchanged_phrases:
                if tokens[i] in string.punctuation:
                    candidate_idxs[i] = 0
                if ' '.join(tokens[i:i + len(splited_unchanged_phrase)]).lower() == ' '.join(
                        splited_unchanged_phrase):
                    candidate_idxs[i:i + len(splited_unchanged_phrase)] = 0
        candidate_idxs = [i for (i, x) in enumerate(candidate_idxs) if x == 1]
        # candidate_idxs=[i for i in range(n) if tokens[i] not in string.punctuation and tokens[i].lower() not in unchanged_word_list]
        idxs_should_be_changed = [i for i in range(n) if tokens[i].lower() in changed_word_list]
        n = len(candidate_idxs)
        indices = sorted(list(set(random.sample(candidate_idxs, int(n * mask_ratio)) + idxs_should_be_changed)))
    if at_least_one == True and len(indices) == 0:
        indices = sorted(random.sample(range(n), 1))
    masked_src, masked_tgt = "", []
    masked_list, masked_out = [], ''
    for i, idx in enumerate(indices):
        if i == 0 or idx != indices[i - 1] + 1:
            masked_tgt.append("")
        masked_tgt[-1] += " " + tokens[idx]
        tokens[idx] = "[MASK]"
    for i, token in enumerate(tokens):
        if i != 0 and token == "[MASK]" and tokens[i - 1] == "[MASK]":
            continue
        if token == "[MASK]":
            masked_src += " " + substitute_verbalizers[cnt]
            masked_list.append(substitute_verbalizers[cnt])
            cnt += 1
        else:
            masked_src += " " + token
    assert len(masked_tgt)==len(masked_list)
    for msk, word in zip(masked_list, masked_tgt):
        masked_out += " " + msk +" " + word
    masked_out += ' '+'</s>'
    return masked_src.strip(), masked_tgt, masked_out.strip()

def nlg_data_wrapper(config, dataset):
    encoder_input_ids, encoder_mask = process_tensor([d[0] for d in dataset], 0, output_mask=True)
    decoder_input_ids, decoder_mask = process_tensor([d[1] for d in dataset], 0, output_mask=True)
    decoder_input_ids[decoder_mask == 0] = -100
    gt_y = [d[2] for d in dataset]
    gt_x = [d[3] for d in dataset]
    if len(dataset[0]) == 5:
        task_index = torch.tensor([d[4] for d in dataset]).long()
    else:
        task_index = torch.tensor([0 for _ in range(len(dataset))]).long()

    return {"task_index": task_index, "encoder_input_ids": encoder_input_ids, "encoder_mask": encoder_mask, "decoder_input_ids": decoder_input_ids, "gt_y": gt_y, "gt_x": gt_x}

def nlg_get_data_loader(config, dataset, batch_size, shuffle=False):
    collate_fn = lambda d: nlg_data_wrapper(config, d)
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=shuffle
    )

def process_tensor(tensor_list, last_dim, output_mask=False):
    tensor_len = [d.shape[0] for d in tensor_list]
    tensor_max_lenth = max(tensor_len)
    d_type = tensor_list[0].dtype
    if last_dim > 0:
        tensor_np = np.zeros((len(tensor_list), tensor_max_lenth, last_dim), dtype=d_type)
    else:
        tensor_np = np.zeros((len(tensor_list), tensor_max_lenth), dtype=d_type)
    mask_np = np.zeros((len(tensor_list), tensor_max_lenth), dtype=np.float32)
    for i, (d, l) in enumerate(zip(tensor_list, tensor_len)):
        if l > 0:
            tensor_np[i, :l] = d
            mask_np[i, :l] = 1
    if output_mask:
        return torch.from_numpy(tensor_np), torch.from_numpy(mask_np)
    else:
        return torch.from_numpy(tensor_np)