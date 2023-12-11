import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parameter import Parameter


class Dip_l(nn.Module):
    # Dipole - Location-based Attention
    def __init__(self, input_dim, output_dim, hidden_dim, bi_direction=False, activation='sigmoid'):
        super(Dip_l, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction

        self.bi = 1
        if self.bi_direction:
            self.bi = 2
        self.rnn = nn.GRU(input_dim, self.hidden_dim, 1, bidirectional=self.bi_direction)  # 双向RNN

        self.attention_t = nn.Linear(self.hidden_dim * self.bi, 1)
        self.a_t_softmax = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(self.hidden_dim * 2 * self.bi, self.output_dim)
        if activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Softmax(1)

    def forward(self, x, verbose=False):
        self.rnn.flatten_parameters()
        rnn_in = x.permute(1, 0, 2) # 这里我猜测本来是以 batch, seq_length, input_size方式组织的x，所以要变成没有batch_first的输入
        rnn_out, h = self.rnn(rnn_in)
        rnn_out = rnn_out.permute(1, 0, 2)
        out = rnn_out[:, :-1, :]
        if verbose:
            print("out", out.shape, '\n', out)

        location_attention = self.attention_t(out)
        a_t_softmax_out = self.a_t_softmax(location_attention)
        context = torch.mul(a_t_softmax_out, out)
        sum_context = context.sum(dim=1)

        if verbose:
            print("location_attention", location_attention.shape, '\n', location_attention)
            print("a_t_softmax_out", a_t_softmax_out.shape, '\n', a_t_softmax_out)
            print("context", context.shape, '\n', context)

        concat_input = torch.cat([rnn_out[:, -1, :], sum_context], 1)
        output = self.linear1(concat_input)
        output = self.out_activation(output)
        return output


class Dip_g(nn.Module):
    # Dipole - General Attention
    def __init__(self, input_dim, output_dim, hidden_dim, bi_direction=False, activation='sigmoid'):
        super(Dip_g, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction

        self.bi = 1
        if self.bi_direction:
            self.bi = 2
        self.rnn = nn.GRU(input_dim, self.hidden_dim, 1, bidirectional=self.bi_direction)

        self.attention_t_w = Parameter(
            torch.randn(self.hidden_dim * self.bi, self.hidden_dim * self.bi, requires_grad=True).float())
        self.a_t_softmax = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(self.hidden_dim * 2 * self.bi, self.output_dim)
        if activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Softmax(1)
        # nn.init.kaiming_uniform_(self.attention_t_w,a=np.sqrt(5))

    def forward(self, x, verbose=False):
        self.rnn.flatten_parameters()
        rnn_in = x.permute(1, 0, 2)
        rnn_out, h = self.rnn(rnn_in)
        rnn_out = rnn_out.permute(1, 0, 2)
        out = rnn_out[:, :-1, :]
        last_out = rnn_out[:, -1, :]
        if verbose:
            print("out", out.shape, '\n', out)

        general_attention = torch.matmul(last_out, self.attention_t_w)
        general_attention = torch.matmul(out, general_attention.unsqueeze(-1))
        a_t_softmax_out = self.a_t_softmax(general_attention)
        context = torch.mul(a_t_softmax_out, out)
        sum_context = context.sum(dim=1)

        if verbose:
            print("general_attention", general_attention.shape, '\n', general_attention)
            print("a_t_softmax_out", a_t_softmax_out.shape, '\n', a_t_softmax_out)
            print("context", context.shape, '\n', context)

        concat_input = torch.cat([last_out, sum_context], 1)

        output = self.linear1(concat_input)
        output = self.out_activation(output)
        return output


class Dip_c(nn.Module):
    # Dipole - concatenation-based Attention
    def __init__(self, input_dim, output_dim, hidden_dim, max_timesteps, bi_direction=False, activation='sigmoid'):
        super(Dip_c, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction
        self.max_timesteps = max_timesteps

        self.bi = 1
        if self.bi_direction:
            self.bi = 2
        self.rnn = nn.GRU(input_dim, self.hidden_dim, 1, bidirectional=self.bi_direction)

        self.latent = 16
        self.attention_t = nn.Linear(self.hidden_dim * 2 * self.bi, self.latent, bias=False)
        self.attention_v = nn.Linear(self.latent, 1, bias=False)
        self.a_t_softmax = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(self.hidden_dim * 2 * self.bi, self.output_dim)
        if activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Softmax(1)

    def forward(self, x, verbose=False):
        self.rnn.flatten_parameters()
        rnn_in = x.permute(1, 0, 2)
        rnn_out, h = self.rnn(rnn_in)
        rnn_out = rnn_out.permute(1, 0, 2)
        out = rnn_out[:, :-1, :]
        last_out = rnn_out[:, -1, :]

        re_ht = last_out.unsqueeze(1).repeat(1, self.max_timesteps - 1, 1)
        concat_input = torch.cat([re_ht, out], 2)
        concatenation_attention = self.attention_t(concat_input)
        concatenation_attention = torch.tanh(concatenation_attention)
        concatenation_attention = self.attention_v(concatenation_attention)
        a_t_softmax_out = self.a_t_softmax(concatenation_attention)
        self.a_t_softmax_out = a_t_softmax_out
        context = torch.mul(a_t_softmax_out, out)
        sum_context = context.sum(dim=1)

        if verbose:
            print("re_ht", re_ht.shape, '\n', re_ht)
            print("concat_input", concat_input.shape, '\n', concat_input)
            print("concatenation_attention", concatenation_attention.shape, '\n', concatenation_attention)
            print("a_t_softmax_out", a_t_softmax_out.shape, '\n', a_t_softmax_out)
            print("context", context.shape, '\n', context)
            print("sum_context", sum_context.shape, '\n', sum_context)

        concat_input = torch.cat([last_out, sum_context], 1)

        output = self.linear1(concat_input)
        output = self.out_activation(output)
        return output


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