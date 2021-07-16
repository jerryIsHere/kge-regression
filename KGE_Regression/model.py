from transformers.models.roberta.modeling_roberta import *
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import L1Loss
from torch.nn import MSELoss


class RobertaForMLPEmbeddingRegression(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = L1Loss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaEmbeddingMappingHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(config.hidden_size, config.hidden_size, 1)
        self.conv2 = nn.Conv1d(config.hidden_size, config.num_labels, 1)
        self.conv3 = nn.Conv1d(config.num_labels, config.num_labels, 1)

        self.atten1 = nn.Conv1d(config.hidden_size, 384, 1)
        self.atten2 = nn.Conv1d(384, 192, 1)
        self.atten3 = nn.Conv1d(192, 1, 1)

        self.fcn1_1 = nn.Conv1d(config.hidden_size, config.hidden_size, 1)
        self.fcn1_2 = nn.Conv1d(config.hidden_size, config.num_labels, 1)
        self.fcn1_3 = nn.Conv1d(config.num_labels, config.num_labels, 1)

        self.fcn2_1 = nn.Conv1d(config.num_labels, config.num_labels, 1)
        self.fcn2_2 = nn.Conv1d(config.num_labels, config.num_labels, 1)
        self.fcn2_3 = nn.Conv1d(config.num_labels, config.num_labels, 1)

        self.fcn3_1 = nn.Conv1d(config.num_labels, config.num_labels, 1)
        self.fcn3_2 = nn.Conv1d(config.num_labels, config.num_labels, 1)
        self.fcn3_3 = nn.Conv1d(config.num_labels, config.num_labels, 1)

    def forward(self, features, **kwargs):
        x = features.permute(0, 2, 1)[:, :, 1:]
        a = features.permute(0, 2, 1)[:, :, 1:]
        s = features.permute(0, 2, 1)[:, :, 0:1]

        a = self.atten1(a)
        a = torch.nn.ReLU()(a)
        a = self.atten2(a)
        a = torch.nn.ReLU()(a)
        a = self.atten3(a)
        a = torch.nn.Tanh()(a)

        x = self.conv1(x)
        x = torch.nn.ReLU()(x)
        x = self.conv2(x)
        x = torch.nn.ReLU()(x)
        x = self.conv3(x)
        x = torch.nn.ReLU()(x)

        s = self.fcn1_1(s)
        s = torch.nn.ReLU()(s)
        s = self.fcn1_2(s)
        s = torch.nn.ReLU()(s)
        s = self.fcn1_3(s)
        s = torch.nn.ReLU()(s)

        r = self.fcn2_1(s)
        r = torch.nn.ReLU()(r)
        r = self.fcn2_2(r)
        r = torch.nn.ReLU()(r)
        r = self.fcn2_3(r)
        s = s + torch.nn.ReLU()(r)

        r = self.fcn3_1(s)
        r = torch.nn.ReLU()(r)
        r = self.fcn3_2(r)
        r = torch.nn.ReLU()(r)
        r = self.fcn3_3(r)
        s = s + torch.nn.ReLU()(r)

        return s + torch.sum(x * a, dim=2, keepdim=True), torch.sum(torch.abs((a)))


class RobertaForEmbeddingRegression(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def set_attention_loss_weight(self, attention_loss_weight):
        self.attention_loss_weight = attention_loss_weight

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.mapper = RobertaEmbeddingMappingHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits, l1_attention_loss = self.mapper(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss = (
                torch.sqrt(loss_fct(logits.view(-1), labels.view(-1)))
                + self.attention_loss_weight * l1_attention_loss
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
