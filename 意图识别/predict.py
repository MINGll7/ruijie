from eval import evaluate
from data import *
from model import *
from paddlenlp.transformers import ErnieTokenizer, ErnieModel, LinearDecayWithWarmup


"""实现一个模型预测的函数"""
model_name = "ernie-1.0"

slot2id = Data_process.slot2id
intent2id = Data_process.intent2id
train_ds = Data_process.train_ds
dev_ds = Data_process.dev_ds
id2intent = Data_process.id2intent
id2slot = Data_process.id2slot
intent_weight = Data_process.process_in_weight(intent2id, train_ds)
batch_sampler = Data_process.process_batch(slot2id, intent2id, train_ds=train_ds, dev_ds=dev_ds)
train_batch_sampler, dev_batch_sampler, train_loader, dev_loader = batch_sampler
tokenizer = ErnieTokenizer.from_pretrained(model_name)
max_seq_len = 512


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

if __name__ == '__main__':
    input_text = "eap202带机数多少"
    predict(input_text, joint_model, tokenizer, id2intent, id2slot)