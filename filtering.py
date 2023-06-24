import os
import logging
import argparse
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from xmc.dataset import MDataset, createDataCSV, createDACSV
from filter.model import LightXML
from xmc.utils import time_since
from filter.get_data import get_train_examples, save_the_new
from filter.get_score import get_overall_score, get_con_score


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=7, help="gpu id")

    #da
    parser.add_argument('--da_name', default="t5p", type=str)
    parser.add_argument('--percentage', type=int, default=80,  help="")
    parser.add_argument("--aug_num", type=int, default=2)

    #pretrained
    parser.add_argument('--pretrained', type=bool, default=False, #True False
                        help="use pretrained LSFL model")
    parser.add_argument("--pretrained_path", default="/data/pengyu/EUR-Lex/model/bert-base_20220926-161453.pth", type=str,
                        help="path of pretrained LSFL model")
    #/data/pengyu/EUR-Lex/model/bert-base_20220926-161453.pth
    #/data/pengyu/AmazonCat-13K/model/bert-base_20230604-180904.pth
    #/data/pengyu/EUR-Lex/model/bert-base_20230531-234550.pth

    #dataset
    parser.add_argument("--data_dir", default="/data/pengyu/EUR-Lex", type=str,#EUR-Lex Wiki10-31K AmazonCat-13K
                        help="The input data dir.")
    parser.add_argument("--train_texts", default="train_texts.txt", type=str,
                        help="")
    parser.add_argument("--train_labels", default="train_labels.txt", type=str,
                        help="")
    parser.add_argument("--test_texts", default="test_texts.txt", type=str,
                        help="")
    parser.add_argument("--test_labels", default="test_labels.txt", type=str,
                        help="")
    parser.add_argument('--valid_size', type=int, default=200,
                        help="size of validation set")
    parser.add_argument('--max_len', type=int, default=512, #512
                        help="")

    #bert
    parser.add_argument('--bert_name', type=str, required=False, default='bert-base')
    parser.add_argument("--bert_path", default="/data/pengyu/bert-base-uncased", type=str,
                        help="")
    parser.add_argument("--roberta_path", default="/data/pengyu/roberta-base", type=str,
                        help="")
    parser.add_argument("--xlnet_path", default="/data/pengyu/xlnet-base-cased", type=str,#xlnet-base-cased
                        help="")
    parser.add_argument('--apex', type=bool, default=False,  # True False
                        help="")

    #training
    parser.add_argument('--batch', type=int, required=False, default=16)#16
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--swa_warmup', type=int, required=False, default=10)
    parser.add_argument('--swa_step', type=int, required=False, default=200)

    #debug
    parser.add_argument('--sample_num', type=int, default=999999999,
                        help="sample number")
    parser.add_argument('--cut_length', type=int, default=500,
                        help="sample number")

    #lightxml
    parser.add_argument('--update_count', type=int, required=False, default=1)
    parser.add_argument('--eval_step', type=int, required=False, default=20000)
    args = parser.parse_args()

    #process args
    args.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    args.model_dir = os.path.join(args.data_dir, 'model')
    os.makedirs(args.model_dir, exist_ok=True)
    args.model_path = os.path.join(args.model_dir, "%s_%s.pth" % (args.bert_name, args.timemark))
    args.model2_path = os.path.join(args.model_dir, "%s_2_%s.pth" % (args.bert_name, args.timemark))
    args.dataset = args.data_dir.split('/')[-1]
    args.da_train_texts = "da/%s_train_texts.txt"%args.da_name
    args.da_train_labels = "da/%s_train_labels.txt"%args.da_name
    args.filtered_train_texts = "da/filtered_%s_train_texts.txt"%args.da_name
    args.filtered_train_labels = "da/filtered_%s_train_labels.txt"%args.da_name
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in args.__dict__.items()]

    #for reproduce
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    workflow(args)



def workflow(args):

    # Dataset
    start_time = time.time()
    logger.info('Data Loading')
    df, label_map = createDataCSV(args)
    args.label_size = len(label_map)
    train_df = df[df['dataType'] == 'train']

    #deal with da times
    record = []
    for i in range(args.aug_num):
        record.append(train_df)

    n_train_df = pd.concat(record, axis=0)
    logger.info(f'origin size {len(n_train_df)}')
    logger.info(f'label size %d' % (len(label_map)))
    logger.info('Time for loading the data: %s' %time_since(start_time))

    # Model
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] ='%d'%args.gpuid
    args.device = torch.device('cuda:0')
    model = LightXML(n_labels=len(label_map), args=args,update_count=args.update_count,
                     use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step)
    model = model.to(args.device)

    # Data Loader
    logger.info('Train')
    tokenizer = model.get_tokenizer()
    train_d = MDataset(n_train_df, 'train', tokenizer, label_map, args.max_len)
    origin_loader = DataLoader(train_d, batch_size=args.batch, num_workers=2, shuffle=False)

    # Data Loader DA
    logger.info('DA Loading')
    df_da = createDACSV(args)
    logger.info(f'augment size {len(df_da)}')
    train_d_da = MDataset(df_da, 'train', tokenizer, label_map, args.max_len)
    augment_loader_da = DataLoader(train_d_da, batch_size=args.batch, num_workers=2, shuffle=False)
    logger.info('Time for loading DA: %s' % time_since(start_time))

    #pre-trained
    args.model_path = args.pretrained_path
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    #Get Score
    logger.info('Get Score of Diversity')
    start_time = time.time()
    div_score = model.one_epoch(0, augment_loader_da, None, mode='diversity')
    logger.info(f'div_score length: {len(div_score)}')
    logger.info('Time for div_score: %s' %time_since(start_time))
    logger.info('Get Score of Consistence')
    start_time = time.time()
    pred_by_origin, _ = model.one_epoch(0, origin_loader, None, mode='test')
    pred_by_augment, labels = model.one_epoch(0, augment_loader_da, None, mode='test')
    con_score = get_con_score(pred_by_origin, pred_by_augment, labels)
    logger.info(f'con_score length: {len(con_score)}')
    logger.info('Time for con_score: %s' %time_since(start_time))
    #overall
    overall_score = get_overall_score(div_score, con_score)
    threshold = np.percentile(overall_score, args.percentage)
    overall_one_hot = [1 if x >= threshold else 0 for x in overall_score]

    #filter
    train_texts, train_labels = get_train_examples(args)
    examples = zip(train_texts, train_labels)
    new_docs = []
    new_labels = []
    for i, (doc, label) in enumerate(examples):
        score = overall_score[i]
        if score==1:
            new_docs.append(doc)
            new_labels.append(label)

    #save data
    save_the_new(new_docs, new_labels, args)


if __name__ == '__main__':
    main()