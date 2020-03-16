# coding=utf-8
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import models
from torch.autograd import Variable
import torch.nn.functional as F


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        _, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def get_representations(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        encoded_layers, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=True,
        )
        pooled_output = self.dropout(pooled_output)
        return encoded_layers, pooled_output

    def get_linear_weights():
        return self.classifier.state_dict()


class BertForSeq2Seq(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSeq2Seq, self).__init__(config)
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.decoder_hidden_size = 768
        self.num_layers = 1
        # self.hidden_linear = nn.Linear(self.hidden_size,
        #                               self.decoder_hidden_size)
        self.context_linear = nn.Linear(
            self.hidden_size,
            self.decoder_hidden_size,
        )
        self.decoder = models.rnn.rnn_decoder(
            num_labels,
            self.decoder_hidden_size,
            num_layers=self.num_layers,
            score_fn="dot",
        )
        self.apply(self.init_bert_weights)
        self.criterion = models.criterion(
            tgt_vocab_size=num_labels,
            use_cuda=True,
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        input_ids,
        labels,
        token_type_ids=None,
        attention_mask=None,
    ):
        encoder_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
        )
        pooled_output = self.dropout(pooled_output)
        h = pooled_output.view(1, -1, self.decoder_hidden_size)
        c = self.context_linear(pooled_output).view(
            1,
            -1,
            self.decoder_hidden_size,
        )
        decoder_outputs, final_state = self.decoder(
            labels[:-1],
            (h, c),
            encoder_output,
        )
        return decoder_outputs, labels[1:]

    def compute_loss(self, hidden_outputs, targets):
        return models.cross_entropy_loss(
            hidden_outputs,
            self.decoder,
            targets,
            self.criterion,
        )

    @staticmethod
    def max_pool(p, q):
        return [
            [max(p[i][j], q[i][j]) for j in range(len(p[i]))] for i in range(len(p))
        ]

    def beam_sample(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        beam_size=1,
    ):
        batch_size = input_ids.size(0)
        encoder_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
        )
        pooled_output = self.dropout(pooled_output)
        h = pooled_output.view(1, -1, self.hidden_size)
        c = self.context_linear(pooled_output).view(1, -1, self.hidden_size)
        encState = (h, c)
        def rvar(a):
            with torch.no_grad():
                return a.repeat(1, beam_size, 1)

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        contexts = rvar(encoder_output.data.transpose(0, 1)).transpose(0, 1)
        decState = (rvar(encState[0].data), rvar(encState[1].data))

        beam = [models.Beam(beam_size, n_best=1, cuda=True) for _ in range(batch_size)]

        mask = None
        soft_score = None
        max_tgt_len = 9

        final_probs = [[0.0] * self.num_labels] * batch_size
        for i in range(max_tgt_len):

            if all((b.done() for b in beam)):
                break
            with torch.no_grad():
                inp = (
                    torch.stack([b.getCurrentState() for b in beam])
                    .t()
                    .contiguous()
                    .view(-1)
                )

            output, decState, attn = self.decoder.sample_one(
                inp,
                soft_score,
                decState,
                contexts,
                mask,
            )
            soft_score = F.softmax(output, dim=-1)
            predicted = output.max(1)[1]
            for i in range(1):
                final_probs = self.max_pool(
                    final_probs,
                    soft_score[batch_size * i : batch_size * (i + 1)],
                )

            if mask is None:
                mask = predicted.unsqueeze(1).long()
            else:
                mask = torch.cat((mask, predicted.unsqueeze(1)), 1)
            output = unbottle(self.log_softmax(output))
            # print(output)
            attn = unbottle(attn)
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)
        allHyps, allScores, allAttn = [], [], []
        for b in beam:
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        final_probs = [[prob.item() for prob in preds[4:]] for preds in final_probs]
        return allHyps, allAttn, final_probs
