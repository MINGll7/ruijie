from model import *
from data import *


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

# 训练参数
num_epoch = 5 # 10
learning_rate = 2e-5
weight_decay = 0.01
warmup_proportion = 0.1
max_grad_norm = 1.0
log_step = 50
eval_step = 1000
seed = 1000
slot2id = Data_process.slot2id
intent2id = Data_process.intent2id
train_ds = Data_process.train_ds
dev_ds = Data_process.dev_ds
id2intent = Data_process.id2intent
id2slot = Data_process.id2slot
intent_weight = Data_process.process_in_weight(intent2id, train_ds)
batch_sampler = Data_process.process_batch(slot2id, intent2id,train_ds=train_ds, dev_ds=dev_ds)
train_batch_sampler, dev_batch_sampler, train_loader, dev_loader = batch_sampler

save_path = "./checkpoint"

# envir setting
set_seed(seed)
use_gpu = True if paddle.get_device().startswith("gpu") else False

if use_gpu:
    paddle.set_device("gpu:0")

ernie = ErnieModel.from_pretrained("ernie-1.0")
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

from eval import evaluate

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



if __name__ == '__main__':
    train()
