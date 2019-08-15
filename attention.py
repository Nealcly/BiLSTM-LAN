# -*- coding: utf-8 -*-

'''
Janurary 2018 by Wei Li
liweihfyz@sjtu.edu.cn
https://www.github.cim/leviswind/transformer-pytorch
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from torch.nn.parameter import Parameter


class embedding(nn.Module):

    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        '''Embeds a given Variable.
        Args:
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
        '''
        super(embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = self._backend.Embedding.apply(
            inputs, self.lookup_table, self.padding_idx, None, 2, False, False)  # copied from torch.nn.modules.sparse.py

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-8):
        '''Applies layer normalization.
        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        '''
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features)).cuda()
        self.beta = nn.Parameter(torch.zeros(features)).cuda()

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma.cuda() * (x.cuda() - mean) / (std + self.epsilon) + self.beta.cuda()


class positional_encoding(nn.Module):

    def __init__(self, num_units, zeros_pad=True, scale=True):
        '''Sinusoidal Positional_Encoding.
        Args:
          num_units: Output dimensionality
          zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
          scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
        '''
        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs):
        # inputs: A 2d Tensor with shape of (N, T).
        N, T = inputs.size()[0: 2]

        # First part of the PE function: sin and cos argument
        position_ind = Variable(torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1).long())
        position_enc = torch.Tensor([
            [pos / np.power(10000, 2. * i / self.num_units) for i in range(self.num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a Variable
        lookup_table = Variable(position_enc)

        if self.zeros_pad:
            lookup_table = torch.cat((Variable(torch.zeros(1, self.num_units)),
                                     lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1

        outputs = self._backend.Embedding.apply(
            position_ind, lookup_table, padding_idx, None, 2, False, False)   # copied from torch.nn.modules.sparse.py

        if self.scale:
            outputs = outputs * self.num_units ** 0.5

        return outputs


class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units).cuda(), nn.ReLU().cuda())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units).cuda(), nn.ReLU().cuda())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units).cuda(), nn.ReLU().cuda())

        self.output_dropout = nn.Dropout(p=self.dropout_rate)
        self.normalization = layer_normalization(self.num_units).cuda()

    def forward(self, queries, keys, values,last_layer = False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]

        # Linear projections
        Q = self.Q_proj(queries.cuda()).cuda()  # (N, T_q, C)
        K = self.K_proj(keys.cuda()).cuda()  # (N, T_q, C)
        V = self.V_proj(values.cuda()).cuda()  # (N, T_q, C)

        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        padding = Variable(torch.ones(*outputs.size()).cuda() * (-2 ** 32 + 1))
        condition = key_masks.eq(0.).float()
        outputs = padding.cuda() * condition.cuda() + outputs.cuda() * (1. - condition.cuda())

        # Causality = Future blinding
        # if self.causality:
        #     diag_vals = torch.ones(*outputs[0, :, :].size()).cuda()  # (T_q, T_k)
        #     tril = torch.tril(diag_vals, diagonal=0)  # (T_q, T_k)
        #     # print(tril)
        #     masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N, T_q, T_k)
        #
        #     padding = Variable(torch.ones(*masks.size()).cuda() * (-2 ** 32 + 1))
        #     condition = masks.eq(0.).float()
        #     outputs = padding * condition + outputs * (1. - condition)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs.cuda() * query_masks.cuda()


        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        if last_layer == True:
            #outputs = F.softmax(outputs, dim=-2)
            return outputs

        # Weighted sum
        outputs = torch.bmm(outputs.cuda(), V_.cuda())  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries.cuda()

        # Normalize
        #outputs = self.normalization(outputs)  # (N, T_q, C)
        #return outputs
        return outputs


class feedforward(nn.Module):

    def __init__(self, in_channels, num_units=[2048, 512]):
        '''Point-wise feed forward net.
        Args:
          in_channels: a number of channels of inputs
          num_units: A list of two integers.
        '''
        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units

        # nn.Linear is faster than nn.Conv1d
        self.conv = False
        if self.conv:
            params = {'in_channels': self.in_channels, 'out_channels': self.num_units[0],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv1 = nn.Sequential(nn.Conv1d(**params), nn.ReLU())
            params = {'in_channels': self.num_units[0], 'out_channels': self.num_units[1],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv2 = nn.Conv1d(**params)
        else:
            self.conv1 = nn.Sequential(nn.Linear(self.in_channels, self.num_units[0]), nn.ReLU()).cuda()
            self.conv2 = nn.Linear(self.num_units[0], self.num_units[1]).cuda()
        self.normalization = layer_normalization(self.in_channels)

    def forward(self, inputs):
        if self.conv:
            inputs = inputs.permute(0, 2, 1)
        outputs = self.conv1(inputs.cuda()).cuda()
        outputs = self.conv2(outputs.cuda()).cuda()

        # Residual connection
        outputs += inputs

        # Layer normalization
        if self.conv:
            outputs = self.normalization(outputs.permute(0, 2, 1))
        else:
            outputs = self.normalization(outputs)

        return outputs


class label_smoothing(nn.Module):

    def __init__(self, epsilon=0.1):
        '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
        Args:
            epsilon: Smoothing rate.
        '''
        super(label_smoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        K = inputs.size()[-1]
        return ((1 - self.epsilon) * inputs) + (self.epsilon / K)


if __name__ == '__main__':
    num_units = 512
    inputs = Variable(torch.randn((100, 10)))
    outputs = positional_encoding(num_units)(inputs)
    outputs = multihead_attention(num_units)(outputs, outputs, outputs)
    outputs = feedforward(num_units)(outputs)

    print(outputs)