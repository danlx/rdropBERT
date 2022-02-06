import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from .module import IntentClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        
        # print(input_ids)
        # print(intent_logits.shape)
        # print(intent_label_ids.shape)
        # print(intent_label_ids.view(-1).shape)

        # 1. Intent Softmax
        if intent_label_ids is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            if self.args.do_rdrop:
                intent_loss_fct = nn.KLDivLoss(reduction="batchmean", log_target=True)
                LogSoftmax = nn.LogSoftmax(dim=1)
                logits = LogSoftmax(intent_logits.view(-1, self.num_intent_labels))
                # print(logits.shape)
                # print(logits)
                num = int(logits.shape[0]/2)
                # print(intent_loss_fct(logits[:num], logits[num:]),
                #       intent_loss_fct(logits[num:], logits[:num]))
                # print(intent_loss)
                intent_loss += 2.0*(intent_loss_fct(logits[:num], logits[num:]) + 
                                    intent_loss_fct(logits[num:], logits[:num]))
                # print(intent_loss)

        outputs = ((intent_logits),)

        outputs = (intent_loss,) + outputs

        return outputs  # (loss), logits
