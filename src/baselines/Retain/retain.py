import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math


class Retain(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, bi_direction=False, device=torch.device('cpu:0'),
                 activation="sigmoid"):
        super(Retain, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = input_dim  # 为了保证后续输出格式，这里必须是input_dim
        self.bi_direction = bi_direction
        self.bi = 1  # Retain 默认为单向RNN
        if self.bi_direction:
            self.bi = 2
        # input: (batch_size, seq_length(visit_num), input_size)
        self.alpha_gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bi_direction
        )
        self.beta_gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bi_direction
        )

        self.alpha_li = nn.Linear(self.hidden_dim * self.bi, 1)
        self.beta_li = nn.Linear(self.hidden_dim * self.bi, self.hidden_dim)

        self.output = nn.Linear(self.hidden_dim, self.output_dim)
        if activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Softmax(1)

    def forward(self, x):
        device = self.device
        self.alpha_gru.flatten_parameters()
        self.beta_gru.flatten_parameters()

        rnn_in = x  # input: (batch_size, seq_length(visit_num), input_size)

        # hidden_size = input_size
        g, _ = self.alpha_gru(rnn_in.to(device))  # g: (batch_size, seq_length, bi*input_size)
        h, _ = self.alpha_gru(rnn_in.to(device))  # h: (batch_size, seq_length, bi*input_size)

        # g = g.squeeze(dim=0)  # (seq_length, bi*input_size)
        # h = h.squeeze(dim=0)  # (seq_length, bi*input_size)

        g_li = self.alpha_li(g)  # (batch_size, seq_length, 1)
        h_li = self.beta_li(h)  # (batch_size, seq_length, bi*input_size)

        attn_g = F.softmax(g_li, dim=-1)  # (batch_size, seq_length, 1)
        attn_h = F.tanh(h_li)  # (batch_size, seq_length, bi*input_size)

        c = attn_g * attn_h * (x)  # (batch_size, seq_length, bi*input_size)
        c = torch.sum(c, dim=1)  # (batch_size, bi*input_size)
        # c = torch.sum(c, dim=1).unsqueeze(dim=1)  # (batch_size, 1, bi*input_size)

        output = self.output(c)  # (batch_size, output_dim)
        return self.out_activation(output)
