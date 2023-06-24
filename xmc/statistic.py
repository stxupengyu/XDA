import numpy as np
from tqdm import tqdm
import os

def get_stat(trainloader, label_map, args):
    if os.path.exists(args.stat_path):
        label_stat = np.load(args.stat_path, allow_pickle=True)
    else:
        label_stat = np.zeros(len(label_map))
        for i, data in enumerate(tqdm(trainloader)):
            label = data[-1]
            label_sum = label.sum(axis=0).numpy()
            label_stat = label_stat+ label_sum #[22, 1, 5]
        np.save(args.stat_path, label_stat)
    head_list = label_stat.argsort()[-args.head_num:]  # [1, 2, 0]
    return label_stat, head_list

def get_weight(label_stat, args):
    label_weight = [(max(label_stat)/(i+1))**args.q for i in label_stat]
    label_weight = [i/sum(label_weight)*args.label_size for i in label_weight]
    # label_weight = [(1-args.beta)/(1-args.beta**(ny+1)) for ny in label_stat]
    # mean = np.mean(label_weight)
    # label_weight = [i/mean for i in label_weight]
    return label_weight