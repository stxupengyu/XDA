import os
import logging
import argparse
import time
import random
import numpy as np
import torch
import torch.nn.init as init
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from xmc import train, calibrate, statistic, evaluate
from xmc.dataset import MDataset, createDataCSV, createDACSV
from xmc.model import LightXML
from xmc.utils import time_since

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=7, help="gpu id")

    #da
    parser.add_argument('--da', default=True, action='store_true') #True False
    parser.add_argument('--da_name', default="t5p", type=str)
    parser.add_argument('--head_num', type=int, default=10,
                        help="head to tail threshold")
    parser.add_argument('--stat_path', default="stat_list.npy", type=str)
    parser.add_argument('--adjust', default=True, action='store_true') #True False
    parser.add_argument('--mda', default=True, action='store_true') #True False
    parser.add_argument('--warmup', type=int, default=0,
                        help="begin epoch of adjustment")
    parser.add_argument('--weight', type=float, default=0.3,
                        help="weight of adjustment")
    parser.add_argument('--q', type=float, default=0.5,
                        help="weight of adjustment")
    parser.add_argument('--beta', type=float, default=0.8,
                        help="weight of adjustment")

    #pretrained
    parser.add_argument('--pretrained', type=bool, default=False, #True False
                        help="use pretrained LSFL model")
    parser.add_argument("--pretrained_path", default="/data/pengyu/AmazonCat-13K/model/bert-base_20230604-180904.pth", type=str,
                        help="path of pretrained LSFL model")
    #'/data/pengyu/EUR-Lex/model/bert-base_20220926-161453.pth'
    # "/data/pengyu/AmazonCat-13K/model/bert-base_20230604-180904.pth"
    # /data/pengyu/EUR-Lex/model/bert-base_20230531-234550.pth

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
    parser.add_argument('--epoch', type=int, required=False, default=30)
    parser.add_argument('--batch', type=int, required=False, default=16)#16
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")
    parser.add_argument('--early_stop_tolerance', type=int, default=5,
                        help="early stop of LSFL")
    parser.add_argument('--test_each_epoch', type=bool, default=True,#True False
                        help="test performance on each epoch")
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--swa_warmup', type=int, required=False, default=10)
    parser.add_argument('--swa_step', type=int, required=False, default=200)

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
    args.stat_path = os.path.join(args.model_dir, args.stat_path)
    args.dataset = args.data_dir.split('/')[-1]
    args.da_train_texts = "da/filtered_%s_train_texts.txt"%args.da_name
    args.da_train_labels = "da/filtered_%s_train_labels.txt"%args.da_name
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
    train_df, valid_df = train_test_split(df[df['dataType'] == 'train'], test_size=args.valid_size, random_state=args.seed)
    df.iloc[valid_df.index.values, 2] = 'valid'
    logger.info(f'train size {len(train_df)}')
    logger.info(f'valid size {len(valid_df)}')
    logger.info(f'test size %d'%(len(df[df.dataType =="test"])))
    logger.info(f'label size %d' % (len(label_map)))
    logger.info('Time for loading the data: %s' %time_since(start_time))

    # Model
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] ='%d'%args.gpuid
    args.device = torch.device('cuda:0')
    model = LightXML(n_labels=len(label_map), args=args,update_count=args.update_count,
                     use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step)
    # model = torch.nn.DataParallel(model, device_ids=args.gpuid)
    model = model.to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)  # , eps=1e-8)
    if args.apex:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Data Loader
    logger.info('Train')
    tokenizer = model.get_tokenizer()
    train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len)
    valid_d = MDataset(df, 'valid', tokenizer, label_map, args.max_len)
    test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len)
    trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=2, shuffle=True)
    validloader = DataLoader(valid_d, batch_size=args.batch, num_workers=2, shuffle=True)
    testloader = DataLoader(test_d, batch_size=args.batch, num_workers=1, shuffle=False)
    trainloader_da = None
    if args.da:
        start_time = time.time()
        logger.info('DA Loading')
        df_da = createDACSV(args)
        train_d_da = MDataset(df_da, 'train', tokenizer, label_map, args.max_len)
        trainloader_da = DataLoader(train_d_da, batch_size=args.batch, num_workers=2, shuffle=True)
        logger.info('Time for loading DA: %s' % time_since(start_time))

    # Train
    if args.pretrained==False:
        train.train(model, optimizer, trainloader, validloader, testloader, trainloader_da, args.model_path, args)
        logger.info('Time for training: %s' %time_since(start_time))
        logger.info(f'Best Model Path: {args.model_path}')
    else:
        args.model_path = args.pretrained_path

    # Test
    logger.info('Test')
    start_time = time.time()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    prediction, targets = model.one_epoch(0, testloader, None, mode='test')
    result = evaluate.evaluate_xml(targets, prediction, args)
    logger.info(f'Final Test Result: {result}')
    logger.info('Time for testing: %s' %time_since(start_time))

    # Augmentation
    logger.info('Augmentation')
    start_time = time.time()
    label_stat, head_list = statistic.get_stat(trainloader, label_map, args)
    label_weight = statistic.get_weight(label_stat, args)
    args.head_list = head_list
    args.label_weight = label_weight
    logger.info(f'label_weight, max: {max(args.label_weight): .2f}, mean: {np.mean(args.label_weight): .2f}, min: {min(args.label_weight): .2f}' )
    logger.info('head_num/label_size = %d/%d = %.2f '%(args.head_num, args.label_size, args.head_num/args.label_size))
    logger.info('Time for augmentation: %s' % time_since(start_time))

    # Adjustment
    logger.info('Adjustment')
    start_time = time.time()
    for name, value in model.named_parameters():
        if 'l0' in name:
            value.requires_grad = True
        else:
            value.requires_grad = False
    params = filter(lambda p: p.requires_grad, model.parameters())
    init.xavier_uniform_(model.l0.weight)
    init.normal_(model.l0.bias)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    calibrate.calibrate(model, optimizer, trainloader, validloader, testloader, trainloader_da, args.model2_path, args=args)
    logger.info('Time for retraining: %s' % time_since(start_time))

    # Test again
    logger.info('Test')
    start_time = time.time()
    prediction, targets = model.one_epoch(0, testloader, None, mode='test')
    result = evaluate.evaluate_xml(targets, prediction, args)
    logger.info(f'Final Test Result: {result}')
    logger.info('Time for testing: %s' %time_since(start_time))

if __name__ == '__main__':
    main()