import logging
import argparse
import random
from tqdm import tqdm, trange
import torch
import os
import numpy as np
from t5 import t5_model
from t5.t5_config import Config
from da import data_loader, augment, data_save, gen_aug_T5

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    #gpu
    parser.add_argument('--gpuid', type=int, default=6, help="gpu id")

    #T5 config
    parser.add_argument("--config", default='config/nlg_prefix.yml')
    parser.add_argument("--start_from_checkpoint", default='True')  # True False
    parser.add_argument("--checkpoint", default='/data/pengyu/EUR-Lex/model/t5p_20230609-133620.pth') #/data/pengyu/pretrain_web_page_keyword_t5_short
    parser.add_argument("--config-override", default='none')

    #da
    parser.add_argument("--da_name", default="t5p", type=str)
    parser.add_argument("--max_len", default=500, type=int)
    parser.add_argument("--finetune", default=False, type=bool) #True False
    parser.add_argument("--mask_ratio",default=0.15,type=float)
    parser.add_argument("--aug_type",type=str,default='rand_iter_10')
    parser.add_argument("--do_sample",action="store_true")
    parser.add_argument("--num_beams",type=int,default=1)
    parser.add_argument("--aug_num",type=int,default=2)
    parser.add_argument("--model_name_or_path",type=str,default='/data/pengyu/t5/t5-small')
    parser.add_argument('--batch_size', type=int, default=8,help="batch_size")

    #path
    parser.add_argument("--data_dir", default="/data/pengyu/Wiki10-31K", type=str,#EUR-Lex   Wiki10-31K    AmazonCat-13K
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_texts", default="train_texts.txt", type=str,
                        help="")
    parser.add_argument("--train_labels", default="train_labels.txt", type=str,
                        help="")

    #model
    parser.add_argument("--learning_rate", default=4e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--sample_ratio', type=int, default=7,
                        help="sample ratio")
    parser.add_argument('--temp', type=float, default=1.0,
                        help="temperature")

    #debug
    parser.add_argument('--sample_num', type=int, default=999999999,
                        help="sample number")
    parser.add_argument('--cut_length', type=int, default=500,
                        help="sample number")
    parser.add_argument("--show", default=True, type=bool, #True False
                        help="")

    _A = parser.parse_args()
    _C = Config(_A.config, _A.config_override)

    np.random.seed(_C.random_seed)
    random.seed(_C.random_seed)
    torch.manual_seed(_C.random_seed)
    torch.cuda.manual_seed_all(_C.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    _A.output_dir = os.path.join(_A.data_dir, 'da')
    _A.da_train_texts = "%s_%s" % (_A.da_name, _A.train_texts)
    _A.da_train_labels = "%s_%s" % (_A.da_name, _A.train_labels)
    os.makedirs(_A.output_dir, exist_ok=True)

    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in _A.__dict__.items()]

    workflow(_A, _C)

def workflow(_A, _C):

    #gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % _A.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _C.device = device

    #model
    old_prefix_set_number = _C.prefix_set_number
    if old_prefix_set_number > 1 and _C.load_from_pretrained:
        _C.prefix_set_number = 1
    if _C.enable_full_finetune or _C.enable_adam_opt:
        tokenizer, model = t5_model.get_full_finetune_t5_model(_C)
    else:
        tokenizer, model = t5_model.get_t5_model(_C)

    if _A.start_from_checkpoint is 'True':
        # model.load_state_dict(
        #     torch.load(os.path.join(_A.checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))[
        #         'model'], strict=False)
        model.load_state_dict(torch.load(os.path.join(_A.checkpoint), map_location=torch.device('cpu')))
    t5aug = gen_aug_T5.T5Aug(tokenizer, model)
    gen_blanks_func = t5aug.generate_blanks

    #data
    train_texts, train_labels = data_loader.get_train_examples(_A)
    examples = zip(train_texts, train_labels)

    # augment
    new_docs = []
    new_labels = []
    for doc, label in tqdm(examples):
        new_doc, new_label = aug_with_pattern(doc, label, gen_blanks_func, _A)
        new_docs.append(new_doc)
        new_labels.append(new_label)

    # show
    if _A.show:
        logger.info('show begin')
        for example, ad_example in zip(train_texts, new_docs):
            print('================')
            print('original--', example)
            print('augmented-', ad_example)
            break
        logger.info('show end')

    # save aug data
    data_save.save_the_new(new_docs, new_labels, _A)
    return

def aug_with_pattern(doc, label, gen_blanks_func, args):
    bad_words_ids = [[3], [19794], [22354]] + [[2163], [4273], [465], [150], [1525], [58]]
    args.bad_words_ids = bad_words_ids
    texts_to_be_augmented = []
    tgt_texts = []
    masked_docs = []
    new_labels = []
    for aug_idx in range(args.aug_num):
        new_labels.append(label)
        masked_doc, tgt_text = augment.mask_text(doc, mask_ratio=args.mask_ratio)
        texts_to_be_augmented.append(masked_doc)
        masked_docs.append([masked_doc])
        tgt_texts.append(tgt_text)
    pred_blanks = augment.predict_blanks(texts_to_be_augmented, tgt_texts, gen_blanks_func, args)
    filled_parts = augment.recover_examples_from_blanks(masked_docs, pred_blanks)
    new_docs = augment.postprocess_texts(filled_parts)
    return new_docs, new_labels

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

if __name__ == "__main__":
    main()