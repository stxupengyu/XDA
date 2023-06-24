from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from transformers import RobertaTokenizerFast
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

def get_bert(args):
    if 'roberta' in args.bert_name:
        print('load roberta-base')
        model_config = RobertaConfig.from_pretrained(args.roberta_path)#'roberta-base'
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained(args.roberta_path, config=model_config)
    elif 'xlnet' in args.bert_name:
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased', cache_dir= args.xlnet_path)
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config, cache_dir= args.xlnet_path)
    else:
        print('load bert-base-uncased')
        model_config = BertConfig.from_pretrained(args.bert_path)
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained(args.bert_path, config=model_config)
    return bert

class LightXML(nn.Module):
    def __init__(self, n_labels, args, feature_layers=5, dropout=0.5, update_count=1,
                 candidates_topk=10, use_swa=True, swa_warmup_epoch=10, swa_update_step=200):
        super(LightXML, self).__init__()
        self.use_swa = use_swa
        self.swa_warmup_epoch = swa_warmup_epoch
        self.swa_update_step = swa_update_step
        self.swa_state = {}
        self.update_count = update_count
        self.candidates_topk = candidates_topk
        self.bert = get_bert(args)
        self.args = args
        self.feature_layers, self.drop_out = feature_layers, nn.Dropout(dropout)
        self.l0 = nn.Linear(self.feature_layers * self.bert.config.hidden_size, n_labels)
        self.group_y = None

    def forward(self, input_ids, attention_mask, token_type_ids,labels=None, group_labels=None, candidates=None):
        is_training = labels is not None

        outs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[-1]

        out = torch.cat([outs[-i][:, 0] for i in range(1, self.feature_layers + 1)], dim=-1)
        # print('print(outs[-1].shape)', outs[-1].shape)
        # print('print(outs[-1])', outs[-1])
        # out = torch.cat([torch.mean(outs[-i][:, :], dim=1) for i in range(1, self.feature_layers + 1)], dim=-1) # 16*512*768
        out = self.drop_out(out)
        group_logits = self.l0(out)
        logits = group_logits
        if is_training:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return logits, loss
        else:
            return logits

    def save_model(self, path):
        self.swa_swap_params()
        torch.save(self.state_dict(), path)
        self.swa_swap_params()

    def swa_init(self):
        self.swa_state = {'models_num': 1}
        for n, p in self.named_parameters():
            self.swa_state[n] = p.data.cpu().clone().detach()

    def swa_step(self):
        if 'models_num' not in self.swa_state:
            return
        self.swa_state['models_num'] += 1
        beta = 1.0 / self.swa_state['models_num']
        with torch.no_grad():
            for n, p in self.named_parameters():
                self.swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swa_swap_params(self):
        if 'models_num' not in self.swa_state:
            return
        for n, p in self.named_parameters():
            self.swa_state[n], p.data = self.swa_state[n].cpu(), p.data.cpu()
            self.swa_state[n], p.data = p.data.cpu(), self.swa_state[n].to(self.args.device)

    def get_tokenizer(self):
        if 'roberta' in self.args.bert_name:
            print('load roberta-base tokenizer')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        elif 'xlnet' in self.args.bert_name:
            print('load xlnet-base-cased tokenizer')
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        else:
            print('load bert-base-uncased tokenizer')
            tokenizer = BertTokenizer.from_pretrained(self.args.bert_path, do_lower_case=True)
        return tokenizer

    def get_accuracy(self, candidates, logits, labels):
        scores, indices = torch.topk(logits.detach().cpu(), k=10)
        acc1, acc3, acc5, total = 0, 0, 0, 0
        for i, l in enumerate(labels):
            l = set(np.nonzero(l)[0])
            labels = indices[i, :5].numpy()
            acc1 += len(set([labels[0]]) & l)
            acc3 += len(set(labels[:3]) & l)
            acc5 += len(set(labels[:5]) & l)
            total += 1
        return total, acc1, acc3, acc5

    def one_epoch(self, epoch, dataloader, optimizer, mode='train', rebalance=False, args=None):

        p1, p3, p5 = 0, 0, 0
        total, acc1, acc3, acc5 = 0, 0, 0, 0
        train_loss = 0
        if mode == 'train':
            self.train()
        else:
            self.eval()
        if self.use_swa and epoch == self.swa_warmup_epoch and mode == 'train':
            self.swa_init()
        if self.use_swa and mode == 'eval':
            self.swa_swap_params()

        prediction, target = [], []
        with torch.set_grad_enabled(mode == 'train'):
            for step, data in enumerate(tqdm(dataloader)):

                batch = tuple(t for t in data)
                inputs = {'input_ids': batch[0].to(self.args.device),
                          'attention_mask': batch[1].to(self.args.device),
                          'token_type_ids': batch[2].to(self.args.device)}
                if mode == 'train':
                    inputs['labels'] = batch[3].to(self.args.device)
                outputs = self(**inputs)
                if mode == 'train':
                    if rebalance==True:
                        logits = outputs[0]
                        loss = weighted_loss(logits, inputs['labels'], args)#weighted_loss masked_loss
                    else:
                        loss = outputs[1]
                    loss /= self.update_count
                    train_loss += loss.item()
                    if args.apex == True:
                        from apex import amp
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    if step % self.update_count == 0:
                        optimizer.step()
                        self.zero_grad()
                    if self.use_swa and step % self.swa_update_step == 0:
                        self.swa_step()
                else:
                    logits = outputs
                    if mode == 'eval':
                        labels = batch[3]
                        _total, _acc1, _acc3, _acc5 = self.get_accuracy(None, logits, labels.cpu().numpy())
                        total += _total;
                        acc1 += _acc1;
                        acc3 += _acc3;
                        acc5 += _acc5

                        p1 = acc1 / total
                        p3 = acc3 / total / 3
                        p5 = acc5 / total / 5
                    elif mode == 'test':
                        pre_K = 10
                        scores, pred = torch.topk(logits, pre_K)
                        real = batch[3]
                        prediction.append(pred.detach().cpu())
                        target.append(real.detach().cpu())
                    elif mode == 'diversity':
                        pred = logits
                        labels = batch[3].to(self.args.device)
                        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
                        loss = loss_fn(pred, labels).sum(-1).detach().cpu()
                        prediction.append(loss)

        if self.use_swa and mode == 'eval':
            self.swa_swap_params()

        if mode == 'eval':
            return [round(i*100, 2) for i in [p1, p3, p5]]
        elif mode == 'test':
            return torch.cat(prediction, dim=0).numpy(), torch.cat(target, dim=0).numpy()
        elif mode == 'train':
            return train_loss
        elif mode == 'diversity':
            return torch.cat(prediction, dim=0).numpy()


    def get_con(self, origin_loader, augment_loader_da):
        p1, p3, p5 = 0, 0, 0
        total, acc1, acc3, acc5 = 0, 0, 0, 0
        train_loss = 0
        if mode == 'train':
            self.train()
        else:
            self.eval()
        if self.use_swa and epoch == self.swa_warmup_epoch and mode == 'train':
            self.swa_init()
        if self.use_swa and mode == 'eval':
            self.swa_swap_params()

        prediction, target = [], []
        with torch.set_grad_enabled(mode == 'train'):
            for step, data in enumerate(tqdm(dataloader)):

                batch = tuple(t for t in data)
                inputs = {'input_ids': batch[0].to(self.args.device),
                          'attention_mask': batch[1].to(self.args.device),
                          'token_type_ids': batch[2].to(self.args.device)}
                if mode == 'train':
                    inputs['labels'] = batch[3].to(self.args.device)
                outputs = self(**inputs)
                if mode == 'train':
                    if rebalance==True:
                        logits = outputs[0]
                        loss = weighted_loss(logits, inputs['labels'], args)#weighted_loss masked_loss
                    else:
                        loss = outputs[1]
                    loss /= self.update_count
                    train_loss += loss.item()
                    if args.apex == True:
                        from apex import amp
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    if step % self.update_count == 0:
                        optimizer.step()
                        self.zero_grad()
                    if self.use_swa and step % self.swa_update_step == 0:
                        self.swa_step()
                else:
                    logits = outputs
                    if mode == 'eval':
                        labels = batch[3]
                        _total, _acc1, _acc3, _acc5 = self.get_accuracy(None, logits, labels.cpu().numpy())
                        total += _total;
                        acc1 += _acc1;
                        acc3 += _acc3;
                        acc5 += _acc5

                        p1 = acc1 / total
                        p3 = acc3 / total / 3
                        p5 = acc5 / total / 5
                    elif mode == 'test':
                        pre_K = 10
                        scores, pred = torch.topk(logits, pre_K)
                        real = batch[3]
                        prediction.append(pred.detach().cpu())
                        target.append(real.detach().cpu())
                    elif mode == 'diversity':
                        pred = logits
                        labels = batch[3].to(self.args.device)
                        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
                        loss = loss_fn(pred, labels).sum(-1).detach().cpu()
                        prediction.append(loss)
                        if step>3:
                            break

        if self.use_swa and mode == 'eval':
            self.swa_swap_params()

        if mode == 'eval':
            return [round(i*100, 2) for i in [p1, p3, p5]]
        elif mode == 'test':
            return torch.cat(prediction, dim=0).numpy(), torch.cat(target, dim=0).numpy()
        elif mode == 'train':
            return train_loss
        elif mode == 'diversity':
            return torch.cat(prediction, dim=0).numpy()




def masked_loss(logits, labels, args):
    #label: batch_size*label_size
    #logits: batch_size*label_size
    pred = torch.sigmoid(logits).detach() #without gradient
    for head_idx in args.head_list:#mask operation
        labels[:,head_idx] = pred[:,head_idx]
    criteria = nn.BCEWithLogitsLoss()
    loss = criteria(logits, labels.float())
    return loss

def weighted_loss(logits, labels, args):
    #label: batch_size*label_size
    #logits: batch_size*label_size
    # weights = torch.ones(labels.shape[-1]).to(args.device)
    # for head_idx in head_list:#mask operation
    #     weights[head_idx] = 0
    weights = torch.Tensor(args.label_weight).to(args.device)
    loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float(), weight=weights)
    return args.weight *loss