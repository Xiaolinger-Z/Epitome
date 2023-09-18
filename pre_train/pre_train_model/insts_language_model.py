import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import  MaskedLMOutput
from transformers import BertPreTrainedModel, BertModel

class BERTInstsLM(BertPreTrainedModel):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)

        self.CWP= NextSentencePrediction(config.hidden_size)
        self.DUP = NextSentencePrediction(config.hidden_size)

        self.MF = MaskFilling(config.hidden_size, config.vocab_size)

    def forward(self,
        dfg_input_ids: Optional[torch.Tensor] = None,
        dfg_attention_mask: Optional[torch.Tensor] = None,
        dfg_token_type_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):
        # d_segment_label = [1 for _ in range(len(d))]

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        d_outputs = self.bert(input_ids =dfg_input_ids, token_type_ids = dfg_token_type_ids, attention_mask=dfg_attention_mask,
                      output_hidden_states= output_hidden_states, return_dict = return_dict)
        c_outputs = self.bert(input_ids =input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask,
                      output_hidden_states=output_hidden_states, return_dict=return_dict)


        d_sequence_output = d_outputs[0]
        d_prediction_scores = self.DUP(d_sequence_output)

        c_sequence_output = c_outputs[0]
        c_prediction_scores = self.CWP(c_sequence_output)
        c_mf_scores = self.MF(c_sequence_output)

        if not return_dict:
            d_output = (d_prediction_scores,) + d_outputs[2:]
            c_output = (c_prediction_scores,) + c_outputs[2:]
            cmf_output = (c_mf_scores,) + c_outputs[2:]
            return  ((None,)+d_output), ((None,)+c_output), ((None,)+cmf_output) if labels is not None \
                else d_output, c_output, cmf_output


        d_output = MaskedLMOutput(loss=None, logits=d_prediction_scores, hidden_states=d_outputs.hidden_states,
                                  attentions=d_outputs.attentions,)
        c_output = MaskedLMOutput(loss=None, logits=c_prediction_scores, hidden_states=c_outputs.hidden_states,
                                  attentions=c_outputs.attentions, )
        cmf_output = MaskedLMOutput(loss=None, logits=c_mf_scores, hidden_states=c_outputs.hidden_states,
                                  attentions=c_outputs.attentions, )

        return d_output, c_output, cmf_output

class NextSentencePrediction(nn.Module):
    """
    From NSP task, now used for DUP and CWP
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskFilling(nn.Module):

    def __init__(self, hidden, vocab_size):
        super().__init__()

        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
    
    def forward(self, x):

        lm_logits = self.lm_head(x)
        return lm_logits

