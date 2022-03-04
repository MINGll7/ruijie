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


# intent_dict_path = "./dataset/intent_labels.json"
# slot_dict_path = "./dataset/slot_labels.json"
intent_dict_path = "./dataset/intent_labels(1).json"
slot_dict_path = "./dataset/slot_labels(1).json"
train_path = "./dataset/train.json"
dev_path = "./dataset/test.json"

# load and process data
intent2id, id2intent = load_dict(intent_dict_path)
slot2id, id2slot = load_dict(slot_dict_path)

# train_ds = load_dataset(read, data_path=train_path, lazy=False)
train_ds = load_dataset(read, data_path="./dataset/train(1).json",lazy=False)
print(len(train_ds))
dev_ds = load_dataset(read, data_path=dev_path, lazy=False)
# dev_cnt = 0
# for example in dev_ds:
#     if dev_cnt < 5:
#         print(example)
#         dev_cnt+=1
#     else:
#         print("\n")
#         break

# compute intent weight
intent_weight = [1] * len(intent2id)
cnt = 0
for example in train_ds:
    if cnt < 5:
        print(example)
        cnt+=1
    for intent in example["intent_labels"]:
        intent_weight[intent2id[intent]] += 1
for intent, intent_id in intent2id.items():
    neg_pos = (len(train_ds) - intent_weight[intent_id]) / intent_weight[intent_id]
    intent_weight[intent_id] = np.log10(neg_pos)
intent_weight = paddle.to_tensor(intent_weight)

"""将数据转换成适合输入模型的特征形式，即将文本字符串数据转换成字典id的形式。
这里我们要加载paddleNLP中的ErnieTokenizer，其将帮助我们完成这个字符串到字典id的转换。"""
model_name = "ernie-1.0"
max_seq_len = 512
batch_size = 32

tokenizer = ErnieTokenizer.from_pretrained(model_name)
trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, slot2id=slot2id, intent2id=intent2id,  pad_default_tag="O", max_seq_len=max_seq_len)

train_ds = train_ds.map(trans_func, lazy=False)
dev_ds = dev_ds.map(trans_func, lazy=False)

"""构造DataLoader，该DataLoader将支持以batch的形式将数据进行划分，从而以batch的形式训练相应模型。"""
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    Stack(dtype="float32"),
    Pad(axis=0, pad_val=slot2id["O"], dtype="int64"),
):fn(samples)

"""基于ERNIE实现图1所展示的意图识别和槽位填充功能。具体来讲，我们将处理好的文本数据输入ERNIE模型中，
ERNIE将会对文本的每个token进行编码，产生对应向量序列，然后根据CLS位置的token向量进行意图识别任务，
根据后续的位置文本token向量进行槽位填充任务"""

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddlenlp.transformers import ErniePretrainedModel

class JointModel(paddle.nn.Layer):
    def __init__(self, ernie, num_slots, num_intents, dropout=None):
        super(JointModel, self).__init__()
        self.num_slots = num_slots
        self.num_intents = num_intents

        self.ernie = ernie
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"]) # 作用是，在 training 模式下，基于伯努利分布抽样，以概率 p 对张量 input 的值随机置0；training 模式中，对输出以 1/(1-p) 进行 scaling，而 evaluation 模式中，使用恒等函数；

        self.intent_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"]) # 全连接层，输入与输出都是二维张量，一般形状为 [batch_size, size]
        self.slot_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])

        self.intent_classifier = nn.Linear(self.ernie.config["hidden_size"], self.num_intents)
        self.slot_classifier = nn.Linear(self.ernie.config["hidden_size"], self.num_slots)


    def forward(self, token_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        # 只要在实例化一个对象中传入对应的参数就可以自动调用 forward 函数
        # 利用Python的语言特性，y = model(x)是调用了对象model的__call__方法，而nn.Module把__call__方法实现为类对象的forward函数，所以任意继承了nn.Module的类对象都可以这样简写来调用forward函数。
        """
        调用forward方法的具体流程是：执行y = model(x)时，由于LeNet类继承了Module类，而Module这个基类中定义了__call__方法，所以会执行__call__方法，而__call__方法中调用了forward()方法
        只要定义类型的时候，实现__call__函数，这个类型就成为可调用的。 换句话说，我们可以把这个类型的对象当作函数来使用"""
        sequence_output, pooled_output = self.ernie(token_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)
        # ReLU实现稀疏后的模型能够更好地挖掘相关特征，拟合训练数据。ReLU函数其实是分段线性函数，把所有的负值都变为0，而正值不变，这种操作被成为单侧抑制。
        #不存在梯度消失问题(Vanishing Gradient Problem)，使得模型的收敛速度维持在一个稳定状态。
        # 梯度消失问题：当梯度小于1时，预测值与真实值之间的误差每传播一层会衰减一次，如果在深层模型中使用sigmoid作为激活函数，这种现象尤为明显，将导致模型收敛停滞不前。
        sequence_output = F.relu(self.slot_hidden(self.dropout(sequence_output))) 
        pooled_output = F.relu(self.intent_hidden(self.dropout(pooled_output)))

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits

"""定义模型训练时用到的损失函数
存在意图识别和槽位填充两个loss，在本实践中，我们将两者进行相加作为最终的loss"""
class JointLoss(paddle.nn.Layer):
    def __init__(self, intent_weight=None):
        super(JointLoss, self).__init__()
        self.intent_criterion = paddle.nn.BCEWithLogitsLoss(weight=intent_weight) # 计算输入 logit 和标签 label 间的 binary cross entropy with logits loss 损失。
        self.slot_criterion = paddle.nn.CrossEntropyLoss() # 输入input和标签label间的交叉熵损失 ，它结合了 LogSoftmax 和 NLLLoss 的OP计算，可用于训练一个 n 类分类器

    def forward(self, intent_logits, slot_logits, intent_labels, slot_labels):
        intent_loss = self.intent_criterion(intent_logits, intent_labels)
        slot_loss = self.slot_criterion(slot_logits, slot_labels)
        loss = intent_loss + slot_loss

        return loss

num_epoch = 5 # 10
learning_rate = 2e-5
weight_decay = 0.01
warmup_proportion = 0.1
max_grad_norm = 1.0
log_step = 50
eval_step = 1000
seed = 1000

save_path = "./checkpoint"

# envir setting
set_seed(seed)
use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device("gpu:0")

ernie = ErnieModel.from_pretrained(model_name)
joint_model = JointModel(ernie, len(slot2id), len(intent2id), dropout=0.1)

num_training_steps = len(train_loader) * num_epoch
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
# Warmup是在ResNet论文中提到的一种学习率预热的方法，它在训练开始的时候先选择使用一个较小的学习率，训练了一些epoches或者steps(比如4个epoches,10000steps),再修改为预先设置的学习来进行训练
# Decay是学习率衰减方法，它指定在训练到一定epoches或者steps后，按照线性或者余弦函数等方式，将学习率降低至指定值。一般，使用Warmup and Decay，学习率会遵循从小到大，再减小的规律。
decay_params = [p.name for n, p in joint_model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)
# 优化器的作用就是让这个f中的k和b变成一个具体数值，使用这组具体数值能够让误差（y和yhat的误差）最小
optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=joint_model.parameters(), weight_decay=weight_decay, apply_decay_param_fun=lambda x: x in decay_params, grad_clip=grad_clip)

joint_loss = JointLoss(intent_weight)

intent_metric = MultiLabelClassificationScore(id2intent)
slot_metric = SeqEntityScore(id2slot)


"""定义一个通用的train函数和evaluate函数。在训练过程中，
每隔log_steps步打印一次日志，每隔eval_steps步进行评估一次模型，
并当意图识别或者槽位抽取任务的指标中任意一方效果更好时，我们将会进行保存模型"""
def evaluate(joint_model, data_loader, intent_metric, slot_metric):
    
    joint_model.eval()
    intent_metric.reset()
    slot_metric.reset()
    for idx, batch_data in enumerate(data_loader):
        input_ids, token_type_ids, intent_labels, tag_ids = batch_data
        intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids)
        # count intent metric
        intent_metric.update(pred_labels=intent_logits, real_labels=intent_labels)
        # count slot metric
        slot_pred_labels = slot_logits.argmax(axis=-1)
        slot_metric.update(pred_paths=slot_pred_labels, real_paths=tag_ids)

    intent_results = intent_metric.get_result()
    slot_results = slot_metric.get_result()

    return intent_results, slot_results

def train():
    # start to train joint_model
    global_step, intent_best_f1, slot_best_f1 = 0, 0., 0.
    joint_model.train()
    for epoch in range(1, num_epoch+1):
        for batch_data in train_loader:
            input_ids, token_type_ids, intent_labels, tag_ids = batch_data
            intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids)

            loss = joint_loss(intent_logits, slot_logits, intent_labels, tag_ids)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % log_step == 0:
                print(f"epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}")
            if global_step > 0 and global_step % eval_step == 0:
                intent_results, slot_results = evaluate(joint_model, dev_loader, intent_metric, slot_metric)
                intent_result, slot_result = intent_results["Total"], slot_results["Total"]
                joint_model.train()
                intent_f1, slot_f1 = intent_result["F1"], slot_result["F1"]
                if intent_f1 > intent_best_f1 or slot_f1 > slot_best_f1:
                    paddle.save(joint_model.state_dict(), f"{save_path}/best.pdparams")
                if intent_f1 > intent_best_f1:
                    print(f"intent best F1 performence has been updated: {intent_best_f1:.5f} --> {intent_f1:.5f}")
                    intent_best_f1 = intent_f1
                if slot_f1 > slot_best_f1:
                    print(f"slot best F1 performence has been updated: {slot_best_f1:.5f} --> {slot_f1:.5f}")
                    slot_best_f1 = slot_f1
                print(f'intent evalution result: precision: {intent_result["Precision"]:.5f}, recall: {intent_result["Recall"]:.5f},  F1: {intent_result["F1"]:.5f}, current best {intent_best_f1:.5f}')
                print(f'slot evalution result: precision: {slot_result["Precision"]:.5f}, recall: {slot_result["Recall"]:.5f},  F1: {slot_result["F1"]:.5f}, current best {slot_best_f1:.5f}\n')

            global_step += 1

train()

"""实现一个模型预测的函数"""

# load model
model_path = "./checkpoint/best.pdparams"
 
loaded_state_dict = paddle.load(model_path)
ernie = ErnieModel.from_pretrained(model_name)
joint_model = JointModel(ernie, len(slot2id), len(intent2id), dropout=0.1)
joint_model.load_dict(loaded_state_dict)

def predict(input_text, joint_model, tokenizer, id2intent, id2slot):
    joint_model.eval()

    splited_input_text = list(input_text)
    features = tokenizer(splited_input_text, is_split_into_words=True, max_seq_len=max_seq_len, return_length=True)
    input_ids = paddle.to_tensor(features["input_ids"]).unsqueeze(0) # 在第0维上增加一个维度
    token_type_ids = paddle.to_tensor(features["token_type_ids"]).unsqueeze(0)
    seq_len = features["seq_len"]

    intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids) # 模型返回的概率分布
    # parse intent labels
    intent_labels = [id2intent[idx] for idx, v in enumerate(intent_logits.numpy()[0]) if v > 0]

    # parse slot labels
    slot_pred_labels = slot_logits.argmax(axis=-1).numpy()[0][1:(seq_len)-1]
    slot_labels = []
    for idx in slot_pred_labels:
        slot_label = id2slot[idx]
        if slot_label != "O":
            slot_label = list(id2slot[idx])
            slot_label[1] = "-"
            slot_label = "".join(slot_label)
        slot_labels.append(slot_label)
    slot_entities = get_entities(slot_labels)

    # print result
    if intent_labels:
        print("intents: ", ",".join(intent_labels))
    else:
        print("intents: ", "无")
    for slot_entity in slot_entities:
        entity_name, start, end = slot_entity
        print(f"{entity_name}: ", "".join(splited_input_text[start:end+1]))

input_text = "eap202带机数多少"
predict(input_text, joint_model, tokenizer, id2intent, id2slot)