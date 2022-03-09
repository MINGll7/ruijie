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

