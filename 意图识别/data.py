import paddle
import os
import ast
import argparse
import warnings
import numpy as np
from functools import partial
from seqeval.metrics.sequence_labeling import get_entities
from utils.utils import set_seed
from utils.data import read, load_dict, convert_example_to_feature
from utils.metric import SeqEntityScore, MultiLabelClassificationScore

import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieTokenizer, ErnieModel, LinearDecayWithWarmup
from paddlenlp.data import Stack, Pad, Tuple


class Data_process:

    intent_dict_path = "./dataset/intent_labels.json"
    slot_dict_path = "./dataset/slot_labels.json"
    train_path = "./dataset/train.json"
    dev_path = "./dataset/test.json"
    # load and process data
    intent2id, id2intent = load_dict(intent_dict_path)
    slot2id, id2slot = load_dict(slot_dict_path)

    # train_ds = load_dataset(read, data_path=train_path, lazy=False)
    train_ds = load_dataset(read, data_path="./dataset/train(test).json",lazy=False)
    dev_ds = load_dataset(read, data_path=dev_path, lazy=False)

    def process_in_weight(intent2id, train_ds):

        intent_weight = [1] * len(intent2id)
        # cnt = 0
        for example in train_ds:
            # if cnt < 5:
            #     print(example)
            #     cnt+=1
            for intent in example["intent_labels"]:
                intent_weight[intent2id[intent]] += 1
        for intent, intent_id in intent2id.items():
            neg_pos = (len(train_ds) - intent_weight[intent_id]) / intent_weight[intent_id]
            intent_weight[intent_id] = np.log10(neg_pos)
        intent_weight = paddle.to_tensor(intent_weight)

        return intent_weight

    def process_batch(slot2id, intent2id, train_ds, dev_ds):
        
        """将数据转换成适合输入模型的特征形式，即将文本字符串数据转换成字典id的形式。
        这里我们要加载paddleNLP中的ErnieTokenizer，其将帮助我们完成这个字符串到字典id的转换。"""
        model_name = "ernie-1.0"
        max_seq_len = 512
        batch_size = 32

        tokenizer = ErnieTokenizer.from_pretrained(model_name)
        trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, slot2id=slot2id, intent2id=intent2id,  pad_default_tag="O", max_seq_len=max_seq_len)

        train_map = train_ds.map(trans_func, lazy=False)
        dev_map = dev_ds.map(trans_func, lazy=False)

        """构造DataLoader，该DataLoader将支持以batch的形式将数据进行划分，从而以batch的形式训练相应模型。"""
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            Stack(dtype="float32"),
            Pad(axis=0, pad_val=slot2id["O"], dtype="int64"),
        ):fn(samples)
        train_batch_sampler = paddle.io.BatchSampler(train_map, batch_size=batch_size, shuffle=True) # 用于 paddle.io.DataLoader 中迭代式获取mini-batch的样本下标数组，数组长度与 batch_size 一致
        dev_batch_sampler = paddle.io.BatchSampler(dev_map, batch_size=batch_size, shuffle=False)
        train_loader = paddle.io.DataLoader(dataset=train_map, batch_sampler=train_batch_sampler, collate_fn=batchify_fn, return_list=True) #  返回一个迭代器，该迭代器根据 batch_sampler 指定的顺序迭代返回dataset数据。
        dev_loader = paddle.io.DataLoader(dataset=dev_map, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn, return_list=True)
    
        return train_batch_sampler, dev_batch_sampler, train_loader, dev_loader

