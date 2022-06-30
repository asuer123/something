from __future__ import absolute_import
import os
import pickle
import random
import torch
import json
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch.nn as nn
from torch.optim import Adam
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from transformers import BertConfig, BertModel, BertTokenizer, BertForPreTraining, BertForMaskedLM
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from src.nezha.configuration_nezha import NeZhaConfig
from src.nezha.modeling_nezha import NeZhaForMaskedLM, NeZhaForPreTraining, NeZhaModel
from src.utils import set_seed, generate_vocab, get_wwm_phrase, truncate_sequence
from src.utils import combine_corpus_files, reAssign_bert_config, _preprocessing
from src.arg_settings.pretrain_argparse import get_argparse
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# import wandb

MODEL_CLASSES = {
    "nezha": (NeZhaConfig, NeZhaModel, NeZhaForPreTraining, NeZhaForMaskedLM),
    "bert": (BertConfig, BertModel, BertForPreTraining, BertForMaskedLM)
}


def load_vocab(dict_path, encoding='utf-8'):
    """从bert的词典文件中读取词典
    :return: dict {token: idx}
    """
    token_dict = {}
    with open(dict_path, 'r', encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def load_vocab_list(path, encoding='utf-8'):
    """从bert的词典文件中读取词典
    :return: List
    """
    vocab_list = []
    with open(path, 'r', encoding=encoding) as reader:
        for line in reader:
            line = line.strip()
            vocab_list.append(line)
    return vocab_list


class Trainer:
    def __init__(self, train_dataset, val_dataset=None, model=None, args=None):
        comment = args["use_model"] + "-" + args["log_comment"]
        self.summary_writer = SummaryWriter(log_dir=args["log_path"], comment=comment)
        if not os.path.exists(args["log_path"]):
            os.makedirs(args["log_path"])
        # wandb.init(project=project_name, dir=args["log_path"])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epoch = args['epoch']
        self.batch_size = args['per_gpu_batch_size'] * max(1, args['n_gpu'])
        self.batch_expand_times = args['batch_expand_times']
        self.lr = args['lr']
        self.metric = float('inf')

        self.device = args['device']
        train_sampler = DistributedSampler(train_dataset) if args['local_rank'] != -1 else None
        self.train_sampler = train_sampler
        self.train_dataloader = train_dataset.get_dataloader(shuffle=(train_sampler is None), batch_size=self.batch_size,
                                                             sampler=train_sampler)
        self.val_dataloader = val_dataset.get_dataloader(shuffle=False, batch_size=self.batch_size) if val_dataset else None

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.criterion = nn.CrossEntropyLoss()
        optimizer = Adam(optimizer_grouped_parameters, lr=self.lr)

        total_steps = int(len(train_dataset) * self.epoch / (self.batch_size * self.batch_expand_times))
        # BERT: warmup 10000
        warm_up_step = int(args['warm_up']) if args['warm_up'] >= 1 else int(total_steps * args['warm_up'])
        self.cur_step = 0

        if args.get('scheduler', None) == "cosine":
            num_cycles = args.get('num_cycles', 10)
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=warm_up_step,
                num_cycles=num_cycles,
            )
            print("warm-up/total-step: {}/{}".format(warm_up_step, total_steps))
        elif args.get('scheduler', None) == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=warm_up_step
            )
            print("warm-up/total-step: {}/{}".format(warm_up_step, total_steps))
        else:
            self.scheduler = None
            print("No scheduler used.")

        model = model.to(args['device'])
        self.fp16 = args.get('fp16', False)
        self.scaler = GradScaler() if self.fp16 else None
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        # multi-gpu training (should be after apex fp16 initialization)
        if args['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args['local_rank'] != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['local_rank']],
                                                              output_device=args['local_rank'],
                                                              find_unused_parameters=True)
        self.model = model
        self.optimizer = optimizer

        self.save_name_list = []
        self.save_epoch_list = list(range(20, self.epoch, 20)) if args.get('do_save_epoch', False) else []
        self.early_stop_loss = args.get('early_stop_loss', 0)
        self.args = args

    def train_epoch(self, epoch, fold_idx=None, exp_idx=None):
        self.model.train()
        iterator_bar = tqdm(self.train_dataloader, ncols=100)
        loss_sum = 0.0
        step_num = len(self.train_dataloader)

        ##
        # num_masked_tokens = 0       # 被mask的token数
        # num_total_tokens = 0        # 所有token数
        # num_none_masked_sents = 0   # 被mask的句子数
        # num_total_sents = 0         # 所有句子(=num samples)
        ##

        self.optimizer.zero_grad()
        for step, batch in enumerate(iterator_bar):
            # self.model.train()

            ##
            # labels = batch['labels']
            # attention_mask = batch['attention_mask']
            # bs = attention_mask.shape[0]
            #
            # masked_ids = (labels != -100).long()
            # num_masked_tokens += masked_ids.sum().item()
            # num_total_tokens += attention_mask.sum().item()
            # none_mask_sents = (torch.max(masked_ids, dim=1)[0] == 0).long()
            # num_none_masked_sents += none_mask_sents.sum().item()
            # num_total_sents += bs
            # continue
            ##

            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            with autocast(enabled=self.fp16):
                output = self.model(**batch)
            loss = output[0]
            loss_sum += loss.item()

            self.summary_writer.add_scalar("loss", loss.item(), self.cur_step)
            self.summary_writer.add_scalar("lr", self.scheduler.get_last_lr()[0], self.cur_step)
            # wandb.log({"loss": loss.item()})
            # wandb.log({"lr": self.scheduler.get_last_lr()[0]})

            iterator_bar.set_description(
                "EPOCH[{}] LOSS[{:.5f}]".format(epoch, loss.item(), ))

            if self.args['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.batch_expand_times > 1:
                loss = loss / self.batch_expand_times
            # if self.args['fp16']:
            #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #         scaled_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            self.doBackward(loss)
            if ((step + 1) % self.batch_expand_times) == 0:
                self.doOptimize()
            self.cur_step += 1

        ##
        # print("num masked tokens:{}({:.2f})".format(num_masked_tokens, num_masked_tokens / num_total_tokens))
        # print("num none masked sents:{}({:.2f})".format(num_none_masked_sents, num_none_masked_sents / num_total_sents))
        ##

        iterator_bar.set_description("")
        self.optimizer.zero_grad()

        avg_loss = loss_sum / step_num
        return avg_loss

    def doOptimize(self):
        if self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()

    def doBackward(self, loss):
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def train(self, start_epoch=1, fold_idx=None, exp_idx=None):
        # wandb.watch(self.model)
        self.metric = float('inf')
        self.cur_step = 0
        set_seed(self.args['seed'])
        for epoch in range(start_epoch, self.epoch + start_epoch):
            if self.args['local_rank'] != -1:
                self.train_sampler.set_epoch(epoch)
            avg_loss = self.train_epoch(epoch)
            print("EPOCH[{}] TRAIN INFO: LOSS[{:.5f}]".format(epoch, avg_loss))
            self.summary_writer.add_scalar("avg_loss", avg_loss, self.cur_step)
            # wandb.log({"avg_loss": avg_loss})

            # save pretrained model
            if self.args['local_rank'] == 0 or self.args['local_rank'] == -1:
                if avg_loss < self.metric:  # 保存最优模型
                    self.metric = avg_loss
                    self.save_pretrained(
                        filename="{}_epoch{}_loss{:.5f}".format(self.args['use_model'], epoch, self.metric),
                        save_folder=self.args['output_path']
                    )
                if epoch in self.save_epoch_list:   # 保存指定时间节点的模型
                    self.save_pretrained(
                        filename="{}_epoch{}_loss{:.5f}".format(self.args['use_model'], epoch, self.metric),
                        save_folder=os.path.join(self.args['output_path'], 'save_epoch'),
                        record_on=False
                    )
                if avg_loss < self.early_stop_loss: # 提前中止
                    print("Training early stopped. Final loss:{}.".format(avg_loss))
                    break

    def save_pretrained(self, filename, save_folder="./save", max_save_num=1, record_on=True):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, filename)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_save.save_pretrained(save_path)

        if record_on:
            check_list = []
            for i in self.save_name_list:
                if os.path.exists(i):
                    check_list.append(i)
            if len(check_list) == max_save_num:
                del_file = check_list.pop(0)
                remove_folder(del_file)
            self.save_name_list = check_list
            self.save_name_list.append(save_path)


def remove_folder(path):
    if os.path.isdir(path):
        path_list = os.listdir(path)
        for p in path_list:
            remove_folder(os.path.join(path, p))
        os.rmdir(path)
    elif os.path.isfile(path):
        os.remove(path)


def random_mask(text_tokens, vocab_list, mask='[MASK]', pad='[PAD]'):
    input_tokens, output_tokens = [], []
    rands = np.random.random(len(text_tokens))
    for r, i in zip(rands, text_tokens):
        if r < 0.15 * 0.8:
            input_tokens.append(mask)
            output_tokens.append(i)
        elif r < 0.15 * 0.9:
            input_tokens.append(i)
            output_tokens.append(i)
        elif r < 0.15:
            input_tokens.append(random.choice(vocab_list))
            output_tokens.append(i)
        else:
            input_tokens.append(i)
            output_tokens.append(pad)
    return input_tokens, output_tokens


def phrase_combine(tokens, phrases_dict, max_len_phrase=4):
    if tokens is None:
        return tokens

    tokens_with_phrase = []
    i = 0
    while i < len(tokens):
        phrase = tokens[i]
        k = None
        flag = False
        for k in range(max_len_phrase, 1, -1):  # k = [2, max_len_phrase]
            _phrase = tokens[i:i + k]
            _phrase = " ".join(_phrase)
            if _phrase in phrases_dict:
                phrase = _phrase
                flag = True
                break
        tokens_with_phrase.append(phrase)
        i = i + k if flag else i + 1
    return tokens_with_phrase


def ngram_mask_v2(tokens, max_n_gram, vocab_list,
               masked_lm_prob=0.15, max_predictions_per_seq=5, mask_token='[MASK]', pad_token='[PAD]'):

    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, max_n_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_n_gram + 1)
    pvals /= pvals.sum(keepdims=True)
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        cand_indices.append(i)
    num_to_mask = min(max_predictions_per_seq, int(round(len(tokens) * masked_lm_prob)) )
    random.shuffle(cand_indices)
    masked_idx2token = {}   # key: 被mask的token的idx
                            # value: 原来的token
    for index in cand_indices:
        n = np.random.choice(ngrams, p=pvals)
        if len(masked_idx2token) >= num_to_mask:
            break
        if index in masked_idx2token.keys():
            continue
        if index < len(cand_indices) - (n - 1):
            for i in range(n):
                if len(masked_idx2token) >= num_to_mask:
                    break
                ind = index + i
                if ind in masked_idx2token:
                    continue
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_token = mask_token
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = tokens[ind]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = random.choice(vocab_list)
                masked_idx2token[ind] = tokens[ind]
                tokens[ind] = masked_token

    output_tokens = []
    for i, t in enumerate(tokens):
        output_token = masked_idx2token.get(i, pad_token)
        output_tokens.append(output_token)

    return tokens, output_tokens

def ngram_mask(tokens, max_n_gram, vocab_list,
               masked_lm_prob=0.15, max_predictions_per_seq=5, mask_token='[MASK]', pad_token='[PAD]'):

    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, max_n_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_n_gram + 1)
    pvals /= pvals.sum(keepdims=True)
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        cand_indices.append(i)
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    random.shuffle(cand_indices)
    masked_idx2token = {}   # key: 被mask的token的idx
                            # value: 原来的token
    for index in cand_indices:
        n = np.random.choice(ngrams, p=pvals)
        if len(masked_idx2token) >= num_to_mask:
            break
        if index in masked_idx2token.keys():
            continue
        if index < len(cand_indices) - (n - 1):
            for i in range(n):
                if len(masked_idx2token) >= num_to_mask:
                    break
                ind = index + i
                if ind in masked_idx2token:
                    continue
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_token = mask_token
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = tokens[ind]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = random.choice(vocab_list)
                masked_idx2token[ind] = tokens[ind]
                tokens[ind] = masked_token

    output_tokens = []
    for i, t in enumerate(tokens):
        output_token = masked_idx2token.get(i, pad_token)
        output_tokens.append(output_token)

    return tokens, output_tokens


def wwm_mask(text_tokens, vocab_list, mask_token='[MASK]', pad_token='[PAD]'):
    input_tokens, output_tokens = [], []
    # 随机挑选token
    rands = np.random.random(len(text_tokens))
    for r, i in zip(rands, text_tokens):
        i = i.split(' ')
        n = len(i)  # n>=2
        if r < 0.15 * 0.8:
            # n个mask ，所以为什么text_token里会有这种长一点的呢
            input_tokens.extend([mask_token] * n)
            output_tokens.extend(i)
        elif r < 0.15 * 0.9:
            # 保持不变
            input_tokens.extend(i)
            output_tokens.extend(i)
        elif r < 0.15:
            tmp = [random.choice(vocab_list) for i in range(n)]
            input_tokens.extend(tmp)
            output_tokens.extend(i)
        else:
            input_tokens.extend(i)
            output_tokens.extend([pad_token] * n)
    return input_tokens, output_tokens


class PretrainDataset(data.Dataset):
    def __init__(self, data=None, vocab_path=None,
                 use_next_sentence_label=False, use_single_text=False, max_seq_len=None, phrases=None,
                 mask_mode=None, max_n_gram=-1, max_predictions_per_seq=5):

        self.tokenizer = BertTokenizer(vocab_file=vocab_path)
        self.vocab_list = load_vocab_list(vocab_path)
        self.phrases = phrases

        # self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.not_predict_tag = -100
        self.use_next_sentence_label = use_next_sentence_label
        self.use_single_text = use_single_text
        self.mask_mode = mask_mode
        self.max_n_gram = max_n_gram
        self.max_predictions_per_seq = max_predictions_per_seq
        self.init_data(data, phrases)

    def init_data(self, data, phrases):
        ''' 初始化data
        '''
        self.data = data

        # 加入单句
        if self.use_single_text:
            text_set = set()
            for d in self.data:
                text_set.add(" ".join(d['text1']))
                text_set.add(" ".join(d['text2']))
            for text in text_set:
                text = text.split(" ")
                tmp_dict = {
                    'text1': text,
                    'text2': None,
                    'label': None
                }
                self.data.append(tmp_dict)

        # 截断长文本
        if self.max_seq_len is not None:
            for i, d in enumerate(self.data):
                truncate_sequence(self.max_seq_len - 3, d['text1'], d['text2'])

        # n-gram合并
        if phrases:
            if self.mask_mode == "token-level-ngram":
                pass
            else:
                for i, d in enumerate(self.data):
                    text1, text2 = d['text1'], d['text2']
                    self.data[i]['text1'] = phrase_combine(text1, phrases)
                    self.data[i]['text2'] = phrase_combine(text2, phrases)
        return

    # def tokens2ids(self, input_tokens, unk_idx, pad_idx=None):
    #     input_ids = input_tokens
    #     if self.word2idx is not None:
    #         # input_ids = [self.word2idx.get(token, unk_idx) for token in input_tokens]
    #         input_ids = []
    #         for token in input_tokens:
    #             if pad_idx and token == '[PAD]':
    #                 idx = pad_idx
    #             else:
    #                 idx = self.word2idx.get(token, unk_idx)
    #             input_ids.append(idx)
    #     return input_ids

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0, sampler=None):
        pad_idx = self.tokenizer.convert_tokens_to_ids('[PAD]')
        cls_idx = self.tokenizer.convert_tokens_to_ids('[CLS]')
        sep_idx = self.tokenizer.convert_tokens_to_ids('[SEP]')

        def collate_fn(batch):
            '''
                :return {
                    'input_ids'
                    'labels'
                    'attention_mask'
                    'token_type_ids'
                    'next_sentence_label'
                }
            '''

            collate_data = dict()

            input_ids = []
            token_type_ids = []
            labels = []
            next_sentence_label = []
            for i in batch:
                if i['text2']:
                    i_text1, i_text2 = i['text1'], i['text2']

                    # 随机交换文本位置
                    p = random.random()
                    if p > 0.5:
                        i_text1, i_text2 = i_text2, i_text1  # exchange text1 and text2 with probability:0.5

                    # MASK部分
                    if self.mask_mode == "wwm":
                        assert self.phrases is not None
                        i_text1, i_output1 = wwm_mask(i_text1, vocab_list=self.vocab_list)
                        i_text2, i_output2 = wwm_mask(i_text2, vocab_list=self.vocab_list)
                    elif "ngram" in self.mask_mode:
                        assert self.max_n_gram > 1
                        i_text1, i_output1 = ngram_mask_v2(i_text1, max_n_gram=self.max_n_gram, vocab_list=self.vocab_list,
                                                        max_predictions_per_seq=self.max_predictions_per_seq)
                        i_text2, i_output2 = ngram_mask_v2(i_text2, max_n_gram=self.max_n_gram, vocab_list=self.vocab_list,
                                                        max_predictions_per_seq=self.max_predictions_per_seq)
                    else:
                        # 正常的bert mask策略
                        i_text1, i_output1 = random_mask(text_tokens=i_text1, vocab_list=self.vocab_list)
                        i_text2, i_output2 = random_mask(text_tokens=i_text2, vocab_list=self.vocab_list)

                    i_text1, i_text2 = self.tokenizer.convert_tokens_to_ids(i_text1), \
                                       self.tokenizer.convert_tokens_to_ids(i_text2)
                    i_output1, i_output2 = self.tokenizer.convert_tokens_to_ids(i_output1), \
                                           self.tokenizer.convert_tokens_to_ids(i_output2)

                    for j in range(len(i_output1)):
                        if i_output1[j] == pad_idx:
                            i_output1[j] = self.not_predict_tag
                    for j in range(len(i_output2)):
                        if i_output2[j] == pad_idx:
                            i_output2[j] = self.not_predict_tag

                    tmp_tensor = torch.tensor([cls_idx] + i_text1 + [sep_idx] + i_text2 + [sep_idx])
                    input_ids.append(tmp_tensor)

                    tmp_tensor = torch.tensor([0] * (len(i_text1) + 2) + [1] * (len(i_text2) + 1))
                    token_type_ids.append(tmp_tensor)

                    tmp_tensor = torch.tensor([-100] + i_output1 + [-100] + i_output2 + [-100])
                    labels.append(tmp_tensor)

                    if self.use_next_sentence_label:
                        cls_output = i['label']
                        next_sentence_label.append(cls_output)
                else:
                    # 构造单句输入
                    i_text1 = i['text1']

                    # MASK部分
                    if self.mask_mode == "wwm":
                        assert self.phrases is not None
                        i_text1, i_output1 = wwm_mask(i_text1, vocab_list=self.vocab_list)
                    elif "ngram" in self.mask_mode:
                        assert self.max_n_gram > 1
                        i_text1, i_output1 = ngram_mask(i_text1, max_n_gram=self.max_n_gram, vocab_list=self.vocab_list,
                                                        max_predictions_per_seq=self.max_predictions_per_seq)
                    else:
                        # 正常的bert mask策略
                        i_text1, i_output1 = random_mask(text_tokens=i_text1, vocab_list=self.vocab_list)

                    i_text1 = self.tokenizer.convert_tokens_to_ids(i_text1)
                    i_output1 = self.tokenizer.convert_tokens_to_ids(i_output1)
                    for j in range(len(i_output1)):
                        if i_output1[j] == pad_idx:
                            i_output1[j] = self.not_predict_tag

                    input_ids.append(torch.tensor(i_text1))
                    token_type_ids.append(torch.tensor([0] * len(i_text1)))
                    labels.append(torch.tensor(i_output1))

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
            labels = pad_sequence(labels, batch_first=True, padding_value=self.not_predict_tag)
            attention_mask = (input_ids != pad_idx).long()
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=pad_idx)

            collate_data['input_ids'] = input_ids
            collate_data['attention_mask'] = attention_mask
            collate_data['token_type_ids'] = token_type_ids
            collate_data['labels'] = labels
            if self.use_next_sentence_label:
                collate_data['next_sentence_label'] = torch.tensor(next_sentence_label).long()
            return collate_data

        if sampler:
            return data.DataLoader(
                dataset=self,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                sampler=sampler
            )
        else:
            return data.DataLoader(
                dataset=self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def load_state_dict(state_dict_path):
    state_dict = OrderedDict()
    tmp_sd = torch.load(state_dict_path)
    for k in tmp_sd.keys():
        if 'gamma' in k:
            k_ = k[:-5] + 'weight'
        elif 'beta' in k:
            k_ = k[:-4] + 'bias'
        else:
            k_ = k
        state_dict[k_] = tmp_sd[k]
    return state_dict


def ignore_state_dict(state_dict, ignore_keys):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        ignore = False
        for ignore_key in ignore_keys:
            if ignore_key in k:
                ignore = True
                break
        if not ignore:
            new_state_dict[k] = v
    return new_state_dict


def pretrain(args_dict):
    # ======================================================
    # 并行设置
    if args_dict['local_rank'] == -1 or args_dict['no_cuda']:
        device = torch.device("cuda" if torch.cuda.is_available() and not args_dict['no_cuda'] else "cpu")
        args_dict['n_gpu'] = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args_dict['local_rank'])
        device = torch.device("cuda", args_dict['local_rank'])
        torch.distributed.init_process_group(backend='nccl')
        args_dict['n_gpu'] = 1
    args_dict['device'] = device

    # ======================================================
    # load data and vocab
    train_corpus_path = args_dict['train_corpus_path']
    test_corpus_path = args_dict['test_corpus_path']
    data, _ = _preprocessing(train_corpus_path, with_label=True)
    test_data, _ = _preprocessing(test_corpus_path, with_label=False)
    train_data = data + test_data
    # train_data = train_data[:50]

    vocab_list = load_vocab_list(path=args_dict['vocab_path'])
    vocab_size = len(vocab_list)
    args_dict['vocab_size'] = vocab_size

    # 若使用wwm mask策略，生成基于词频信息的 phrases
    mask_mode = args_dict.get('mask_mode', None)
    if mask_mode == "wwm":
        ngram_phrases_folder = args_dict['ngram_phrases_folder']
        phrases = get_wwm_phrase(ngram_phrases_folder, args_dict['min_freq'])
    else:
        phrases = None

    # ======================================================
    # load model

    # load bert config
    bert_config_path = os.path.join(args_dict['raw_model_path'], 'config.json')
    bert_config, change_dict = reAssign_bert_config(bert_config_path, args_dict)

    config_class, model_class, pretrain_class, mlm_class = MODEL_CLASSES[args_dict['use_model']]

    # init bert config and pretrain_mode
    model_config = config_class(**bert_config)
    if args_dict['pretrain_mode'] == "full":  # "full"=mlm+nsp
        model = pretrain_class(model_config)
        use_next_sentence_label = True
    elif args_dict['pretrain_mode'] == "mlm":
        model = mlm_class(model_config)
        use_next_sentence_label = False
    else:
        raise NotImplemented

    if args_dict['local_rank'] in [0, -1]:
        print(model.config.to_dict())

    # load weights
    if args_dict.get('raw_model_path', False) and not change_dict['hidden_size']:
        p = args_dict['raw_model_path']
        pretrained_model = model_class.from_pretrained(p)
        target_vocab_size = model.bert.embeddings.word_embeddings.num_embeddings
        pretrained_model.resize_token_embeddings(target_vocab_size)
        state_dict = pretrained_model.state_dict()
        if change_dict['max_position_embeddings']:
            state_dict = ignore_state_dict(state_dict, ignore_keys=['positions_encoding'])
        model.bert.load_state_dict(state_dict, strict=False)
    else:
        print("No pretrained weights used.")

    # ======================================================
    if args_dict['local_rank'] in [-1, 0]:
        print("#"*5 + " args of rank[{}]".format(args_dict['local_rank']))
        for k, v in args_dict.items():
            print("{}:[{}]".format(k, v))
        print("#"*10)

    model = model.to(device)
    train_dataset = PretrainDataset(data=train_data, vocab_path=args_dict.get('vocab_path'),
                                    use_next_sentence_label=use_next_sentence_label,
                                    phrases=phrases, max_seq_len=bert_config['max_position_embeddings'],
                                    use_single_text=args_dict.get('use_single_text', False),
                                    mask_mode=args_dict['mask_mode'], max_n_gram=3,
                                    max_predictions_per_seq=args_dict['max_predictions_per_seq'])

    trainer = Trainer(train_dataset, val_dataset=None, model=model, args=args_dict)
    trainer.train(start_epoch=args_dict['start_epoch'])


if __name__ == '__main__':

    args_dict = vars(get_argparse())

    set_seed(args_dict['seed'])
    warm_up = args_dict['warm_up']
    args_dict['warm_up'] = float(warm_up) if ('e' or '.' in warm_up) else int(warm_up)

    pretrain(args_dict)
