import torch
from torch import nn, tanh, sigmoid
from torch.nn.functional import relu
from transformers import AutoModel
from dataloader import num_task, VOCAB, masking, hier, sig, rel

class BertClassifier(nn.Module):
    def __init__(self, params):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, len(VOCAB[i])) for i in range(num_task)])
        self.masking_gate = nn.Linear(2, 1)

        if num_task == 2:
            self.merge_classifier_1 = nn.Linear(len(VOCAB[0])+len(VOCAB[1]), len(VOCAB[0]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(bert_output.pooler_output)

        if num_task == 1:
            logits = [self.classifier[i](bert_output.last_hidden_state) for i in range(num_task)]
        elif num_task == 2 and masking:
            token_level = self.classifier[0](bert_output.last_hidden_state)
            sen_level = self.classifier[1](pooled_output)

            if sig:
                gate = sigmoid(self.masking_gate(sen_level))
            else:
                gate = relu(self.masking_gate(sen_level))
            dup_gate = gate.unsqueeze(1).repeat(1, token_level.size()[1], token_level.size()[2])
            wei_token_level = torch.mul(dup_gate, token_level)

            logits = [wei_token_level, sen_level]
        elif num_task == 2 and hier:
            token_level = self.classifier[0](bert_output.last_hidden_state)
            sen_level = self.classifier[1](pooled_output)
            dup_sen_level = sen_level.repeat(1, token_level.size()[1])
            dup_sen_level = dup_sen_level.view(sen_level.size()[0], -1, sen_level.size()[-1])
            logits = [self.merge_classifier_1(torch.cat((token_level, dup_sen_level),2)), self.classifier[1](pooled_output)]
        elif num_task == 2:
            token_level = self.classifier[0](bert_output.last_hidden_state)
            sen_level = self.classifier[1](pooled_output)
            logits = [token_level, sen_level]

        y_hats = [logits[i].argmax(-1) for i in range(num_task)]

        return logits, y_hats
