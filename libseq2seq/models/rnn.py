import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models

import numpy as np


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class rnn_encoder(nn.Module):
    def __init__(self, config, vocab_size):
        super(rnn_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(
            input_size=config.emb_size,
            hidden_size=config.encoder_hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirec,
        )
        self.config = config

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs)[0]
        # for _ in range(2):
        #    print('===============ENCODER OOUTTPPUUTT:===============')
        # print(outputs.size())
        # print(h.size())
        # print(c.size())
        return outputs, (h, c)


class rnn_decoder(nn.Module):
    def __init__(self, tgt_vocab_size, hidden_size, num_layers, score_fn=None):
        super(rnn_decoder, self).__init__()
        emb_size = hidden_size
        dropout_prob = 0.3
        self.embedding = nn.Embedding(tgt_vocab_size, emb_size)
        self.rnn = StackedLSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob,
        )
        self.score_fn = score_fn

        self.attention = models.global_attention(hidden_size, activation=None)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs, init_state, contexts):
        embs = self.embedding(inputs)
        outputs, state, attns = [], init_state, []
        for emb in embs.split(1):
            output, state = self.rnn(emb.squeeze(0), state)
            output, attn_weights = self.attention(output, contexts)
            output = self.dropout(output)
            outputs += [output]
            attns += [attn_weights]
        outputs = torch.stack(outputs)
        attns = torch.stack(attns)
        return outputs, state

    def compute_score(self, hiddens):
        scores = torch.matmul(hiddens, self.embedding.weight.t())
        return scores

    def sample(self, input, init_state, contexts):
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = 9
        soft_score = None
        mask = None
        for i in range(max_time_step):
            output, state, attn_weights = self.sample_one(
                inputs[i],
                soft_score,
                state,
                contexts,
                mask,
            )
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]
            if mask is None:
                mask = predicted.unsqueeze(1).long()
            else:
                mask = torch.cat((mask, predicted.unsqueeze(1)), 1)

        sample_ids = torch.stack(sample_ids)
        attns = torch.stack(attns)
        return sample_ids, (outputs, attns)

    def sample_one(self, input, soft_score, state, contexts, mask):
        emb = self.embedding(input)
        output, state = self.rnn(emb, state)
        hidden, attn_weigths = self.attention(output, contexts)
        output = self.compute_score(hidden)
        if mask is not None:
            output = output.scatter_(1, mask, -9999999999)
        return output, state, attn_weigths
