import argparse
import logging
import torch
import torch.nn as nn
from t5 import t5_model, utils, prompt_dataset
import numpy as np
from t5.t5_config import Config
import os, sys, math
import random
from tqdm import tqdm
from da import data_loader
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("Prompt Learning")

#gpu
parser.add_argument('--gpuid', type=int, default=5,help="gpu id")

# path
parser.add_argument("--data_dir", default="/data/pengyu/EUR-Lex", type=str) # EUR-Lex   Wiki10-31K    AmazonCat-13K
parser.add_argument("--train_texts", default="train_texts.txt", type=str,help="")
parser.add_argument("--train_labels", default="train_labels.txt", type=str,help="")

# data
parser.add_argument('--sample_num', type=int, default=99999999,help="sample number")
parser.add_argument('--cut_length', type=int, default=1000,help="sample number")
parser.add_argument("--mask_ratio", default=0.15, type=float)
parser.add_argument('--epoch', type=int, default=30,help="sample number")

#T5 config
parser.add_argument("--config", default='config/nlg_prefix.yml')
parser.add_argument("--start_from_checkpoint", default='True') #True False
parser.add_argument("--checkpoint", default='/data/pengyu/pretrain_web_page_keyword_t5_short')
parser.add_argument("--config-override", default='none')

#mode
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', action='store_true')
group.add_argument('--validation', action='store_true')
group.add_argument('--test', action='store_true')

if __name__ == "__main__":

    #parameters
    _A = parser.parse_args()
    _A.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    _A.model_dir = os.path.join(_A.data_dir, 'model')
    os.makedirs(_A.model_dir, exist_ok=True)
    _A.model_path = os.path.join(_A.model_dir, "t5p.pth")
    # _A.model_path = os.path.join(_A.model_dir, "%s_%s.pth" % ('t5p', _A.timemark))
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in _A.__dict__.items()]
    _C = Config(_A.config, _A.config_override)

    np.random.seed(_C.random_seed)
    random.seed(_C.random_seed)
    torch.manual_seed(_C.random_seed)
    torch.cuda.manual_seed_all(_C.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % _A.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _C.device = device

    #model mode
    old_prefix_set_number = _C.prefix_set_number
    if old_prefix_set_number > 1 and _C.load_from_pretrained:
        _C.prefix_set_number = 1
    if _C.enable_full_finetune or _C.enable_adam_opt:
        tokenizer, model = t5_model.get_full_finetune_t5_model(_C)#train all parameter
    else:
        tokenizer, model = t5_model.get_t5_model(_C)#train few parameter   get_full_finetune_t5_model get_t5_model

    if _A.start_from_checkpoint is 'True':
        model.load_state_dict(
            torch.load(os.path.join(_A.checkpoint, 'model-best.pth'), map_location=_C.device)[
                'model'], strict=False)
    model = model.to(_C.device)  # .cpu()

    if old_prefix_set_number > 1 and _C.load_from_pretrained:
        model.update_prefix_embedding(old_prefix_set_number)
        _C.prefix_set_number = old_prefix_set_number

    total_parameter_count = 0
    trainable_parameter_count = 0
    for p in model.parameters():
        total_parameter_count += p.numel()
        if p.requires_grad:
            trainable_parameter_count += p.numel()
    print('Total Parameter Count %d' % total_parameter_count)
    print('Trainable Parameter Count %d' % trainable_parameter_count)

    # Data augmentation Example
    # example = "movie and art I really enjoy <extra_id_0> this <extra_id_1>."#
    # input_ids = tokenizer(example, return_tensors="pt").input_ids.to(_C.device)#.cpu()
    # sequence_ids = model.generate(input_ids)
    # sequences = tokenizer.batch_decode(sequence_ids)
    # print('===augmentation (keywords and mask)===')
    # print(example)
    # print(sequences)

    #data
    train_texts, train_labels = data_loader.get_train_examples(_A)
    train_data = prompt_dataset.NLGMixSenClsDataset(_C, _A, train_texts, tokenizer)
    train_loader = prompt_dataset.nlg_get_data_loader(_C, train_data, _C.batch_size, shuffle=True)
    train_iter = iter(train_loader)
    # for text_i in train_texts:
    #     masked_src, _, masked_tgt  = prompt_dataset.mask_text(text_i)
    #     print('text_i\n', text_i)
    #     print('masked_src\n', masked_src)
    #     print('masked_tgt\n', masked_tgt)
    #     input_ids = tokenizer(masked_src, return_tensors='pt').input_ids
    #     labels = tokenizer(masked_tgt, return_tensors='pt').input_ids
    #     outputs = model(input_ids=input_ids, labels=labels)
    #     loss = outputs.loss

    # model.parallelize()
    if _C.num_training_steps == 0:
        _C.num_training_steps = int(len(train_iter) * _C.max_epoch / _C.gradient_accumulation_steps)
    epoch_num = math.ceil(_C.num_training_steps / _C.checkpoint_every_step)

    if _C.enable_adam_opt:
        optimizer = utils.build_optimizer(_C, model)
    elif _C.enable_full_finetune:
        optimizer = utils.build_t5_finetune_optimizer(_C, model)
    else:
        optimizer = utils.build_t5_optimizer(_C, model)

    eval_every = _C.checkpoint_every_step * _C.gradient_accumulation_steps
    total_step = 0
    best_test_performance = 0
    epoch_num = _A.epoch

    for epoch in range(epoch_num):
        print('EPOCH %d / %d' % (epoch + 1, epoch_num))
        model.train()
        for step in tqdm(range(len(train_iter))):
            try:
                batch = next(train_iter)
            except:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            for n in batch:
                if n not in train_loader.dataset.SKIP_ATTRIBUTES and batch[n] is not None:
                    batch[n] = batch[n].to(device)
            total_step += 1
            outputs = model(
                input_ids=batch['encoder_input_ids'],
                task_ids=batch['task_index'],
                attention_mask=batch['encoder_mask'],
                labels=batch['decoder_input_ids'],
            )
            loss = outputs.loss
            loss = loss
            loss.backward()
            optimizer.step()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            optimizer.zero_grad()
        print("loss %.2f" % loss.item())
        torch.save(model.state_dict(), _A.model_path)
        print('\nmodel saved: %s'%_A.model_path)


    # for epoch in range(epoch_num):
    #     print('EPOCH %d / %d' % (epoch + 1, epoch_num))
    #     run_step = eval_every if total_step + eval_every < _C.num_training_steps * _C.gradient_accumulation_steps else _C.num_training_steps * _C.gradient_accumulation_steps - total_step
    #     model.train()
    #     with tqdm(total=math.ceil(run_step / _C.gradient_accumulation_steps), file=sys.stdout) as pbar:
    #         for step in range(run_step):
    #             try:
    #                 batch = next(train_iter)
    #             except:
    #                 train_iter = iter(train_loader)
    #                 batch = next(train_iter)
    #
    #             for n in batch:
    #                 if n not in train_loader.dataset.SKIP_ATTRIBUTES and batch[n] is not None:
    #                     batch[n] = batch[n].to(device)
    #             total_step += 1
    #             # print(batch)
    #             # print(next(model.parameters()).device)
    #             # outputs = model(
    #             #     input_ids=batch['encoder_input_ids'].cpu(),
    #             #     task_ids=batch['task_index'].cpu(),
    #             #     attention_mask=batch['encoder_mask'].cpu(),
    #             #     labels=batch['decoder_input_ids'].cpu(),
    #             # )
    #             outputs = model(
    #                 input_ids=batch['encoder_input_ids'],
    #                 task_ids=batch['task_index'],
    #                 attention_mask=batch['encoder_mask'],
    #                 labels=batch['decoder_input_ids'],
    #             )
    #             loss = outputs.loss
    #             loss = loss / _C.gradient_accumulation_steps
    #             loss.backward()
    #
    #             if (step + 1) % _C.gradient_accumulation_steps == 0:
    #                 optimizer.step()
    #                 if torch.cuda.is_initialized():
    #                     torch.cuda.synchronize()
    #                 pbar.set_description("loss %.2f" % (loss.item() * _C.gradient_accumulation_steps))
    #                 pbar.update(1)
    #                 optimizer.zero_grad()
    #             torch.save(model.state_dict(), _A.model_path)
    #             print('\nmodel saved: %s'%_A.model_path)
    # model.deparallelize()






