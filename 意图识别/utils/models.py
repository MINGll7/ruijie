import paddle
from paddle import nn
from paddlenlp.transformers import ErniePretrainedModel

class JointErnie(ErniePretrainedModel):
    def __init__(self, ernie, slot_dim, intent_dim, use_history=False, dropout=None):
        super(JointErnie, self).__init__()
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.use_history = use_history

        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        
        self.intent_classifier = nn.Linear(self.ernie.config['hidden_size'], self.intent_num_labels)
        self.slot_classifier = nn.Linear(self.ernie.config['hidden_size'], self.slot_num_labels)

        if self.use_history:
            self.intent_hidden = nn.Linear(2 * self.ernie.config['hidden_size'], self.ernie.config['hidden_size'])
            self.slot_hidden = nn.Linear(2 * self.ernie.config['hidden_size'], self.ernie.config['hidden_size'])
        else:
            self.intent_hidden = nn.Linear(self.ernie.config['hidden_size'], self.ernie.config['hidden_size'])
            self.slot_hidden = nn.Linear(self.ernie.config['hidden_size'], self.ernie.config['hidden_size'])

        self.apply(self.init_weights)

    def forward(self,
                words_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                history_ids=None):
        sequence_output, pooled_output = self.ernie(
            words_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        if self.use_history and (history_ids is not None):
            history_output = self.ernie(history_ids)[1]
            sequence_output = paddle.concat(
                    [history_output.unsqueeze(1).tile(repeat_times=[1, sequence_output.shape[1], 1]),
                    sequence_output], axis=-1)
            pooled_output = paddle.concat([history_output, pooled_output], axis=-1)
        sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))
        pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits