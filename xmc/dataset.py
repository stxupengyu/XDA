import os
import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import tqdm
import warnings
warnings.filterwarnings('ignore')

def createDACSV(args):
    labels = []
    texts = []
    dataType = []
    da_train_texts = os.path.join(args.data_dir, args.da_train_texts)
    da_train_labels = os.path.join(args.data_dir, args.da_train_labels)
    with open(da_train_texts) as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('train')
    with open(da_train_labels) as f:
        for i in tqdm.tqdm(f):
            labels.append(i.replace('\n', ''))
    df_row = {'text': texts, 'label': labels, 'dataType': dataType}
    da_df = pd.DataFrame(df_row)
    return da_df

def createDataCSV(args):
    labels = []
    texts = []
    dataType = []
    label_map = {}

    train_texts = os.path.join(args.data_dir, args.train_texts)
    test_texts = os.path.join(args.data_dir, args.test_texts)
    train_labels = os.path.join(args.data_dir, args.train_labels)
    test_labels = os.path.join(args.data_dir, args.test_labels)
    with open(train_texts) as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('train')

    with open(test_texts) as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('test')

    with open(train_labels) as f:
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))

    with open(test_labels) as f:
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))

    assert len(texts) == len(labels) == len(dataType)

    df_row = {'text': texts, 'label': labels, 'dataType': dataType}

    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k] = i
    df = pd.DataFrame(df_row)

    print('label map', len(label_map))
    return df, label_map

class MDataset(Dataset):
    def __init__(self, df, mode, tokenizer, label_map, max_length,token_type_ids=None, candidates_num=None):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.df, self.n_labels, self.label_map = df[df.dataType == self.mode], len(label_map), label_map
        self.len = len(self.df)
        self.tokenizer, self.max_length= tokenizer, max_length
        self.multi_group = False
        self.token_type_ids = token_type_ids
        self.candidates_num = candidates_num

    def __getitem__(self, idx):
        max_len = self.max_length
        review = self.df.text.values[idx].lower()
        labels = [self.label_map[i] for i in self.df.label.values[idx].split() if i in self.label_map]

        review = ' '.join(review.split()[:max_len])

        text = review
        if self.token_type_ids is not None:
            input_ids = self.token_type_ids[idx]
            if input_ids[-1] == 0:
                input_ids = input_ids[input_ids != 0]
            input_ids = input_ids.tolist()
        elif hasattr(self.tokenizer, 'encode_plus'):
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True,
                max_length=max_len,truncation=True
            )
        else:
            # fast
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True,truncation=True
            ).ids

        if len(input_ids) == 0:
            print('zero string')
            assert 0
        if len(input_ids) > self.max_length:
            input_ids[self.max_length - 1] = input_ids[-1]
            input_ids = input_ids[:self.max_length]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        label_ids = torch.zeros(self.n_labels)
        label_ids = label_ids.scatter(0, torch.tensor(labels),
                                      torch.tensor([1.0 for i in labels]))
        return input_ids, attention_mask, token_type_ids, label_ids

    def __len__(self):
        return self.len