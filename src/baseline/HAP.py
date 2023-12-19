import torch
import numpy as np
import torch.nn.functional as F
from torch import nn as nn


torch.manual_seed(0)
np.random.seed(0)


class AttentionMLP(nn.Module):
    def __init__(self, emb_dim, attn_dim):
        super(AttentionMLP, self).__init__()
        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.Wa = nn.Linear(self.emb_dim * 2, self.attn_dim)
        self.ua = nn.Parameter(torch.randn(self.attn_dim)) 
        self.tanh = nn.Tanh()
    
    def forward(self, e_i, e_j):
        # 两个嵌入向量，计算注意力分数，MLP
        # e_i: (batch_size, visit_num, embed_size)
        # e_j: (batch_size, visit_num, embed_size)
        combined_e = torch.cat([e_i, e_j], dim=-1)  # (batch_size, visit_num, embed*2)
        mlp_output = self.tanh(self.Wa(combined_e))  # (batch_size, visit_num, attn_dim)
        attn_score = torch.matmul(mlp_output, self.ua)  # (batch_size, visit_num)
        return attn_score
    

class HAP(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_ancestors, attn_dim, output_dim, dropout_rate=0.1):
        super(HAP).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
    
        self.embedding = nn.Embedding(2850 + num_ancestors,
                                      embed_dim,
                                      padding_idx=input_dim + num_ancestors)
        
        
        self.attention_mlp = AttentionMLP(embed_dim, attn_dim)

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.out_activation = nn.Sigmoid()
    
    def forward(self, x, p2c, c2p, indicatorx, indicator1, indicator2):
        """
        x: input (batch_size, visit_num, input_size(256))
        p2c: parent2children (batch_size, visit_num, parents_size 256)
        c2p: children2parent (batch_size, visit_num, children_size 256)
        indicator: (batch_size, visit_num, 256[1/0])
        """
        emb_x = self.embedding(x)
        emb_p2c = self.embedding(p2c)
        emb_c2p = self.embedding(c2p)
        # (batch_size, visit_num, 256, embedding_size)

        # 注意力机制
        emb_x = torch.einsum("bvie, bvi->bvie", emb_x, indicatorx)
        emb_p2c = torch.einsum("bvie, bvi->bvie", emb_p2c, indicator1)
        emb_c2p = torch.einsum("bvie, bvi->bvie", emb_c2p, indicator2)
        # (batch_size, visit_num, 256, embedding_size)

        attention_p2c = torch.zeros(emb_p2c.shape[:-1], device=emb_p2c.device)
        attention_c2p = torch.zeros(emb_c2p.shape[:-1], device=emb_c2p.device)
        # (batch_size, visit_num, 256)

        for i in range(emb_p2c.shape[2]):
            attn_scores = self.attention_mlp(emb_x[:, :, i, :], emb_p2c[:, :, i, :])
            attention_p2c[:, :, i] = attn_scores  # (batch_size, visit_num)
        
        attention_p2c = F.softmax(attention_p2c, dim=-1)  # (batch_size, visit_num, 256)

        for i in range(emb_c2p.shape[2]):
            attn_scores = self.attention_mlp(emb_x[:, :, i, :], emb_c2p[:, :, i, :])
            attention_c2p[:, :, i] = attn_scores  # (batch_size, visit_num)
        
        attention_c2p = F.softmax(attention_c2p, dim=-1)  # (batch_size, visit_num, 256)

        # 使用注意力权重求和：
        emb_p2c = torch.einsum('bvla, bvl->bva', emb_x, attention_p2c)
        emb_c2p = torch.einsum('bvla, bvl->bva', emb_x, attention_c2p)
        # (batch_size, visit_num, embed_size)

        emb_weighted = emb_p2c + emb_c2p  # (batch_size, visit_num, embed_size)

        rnn_out, _ = self.rnn(emb_weighted)
        # rnn_out: (batch_size, visit_num, hidden_size)

        output = self.linear(rnn_out)
        # output: (batch_size, visit_num, output_dim)

        output = torch.sum(output, dim=1)  # batch_size, output_dim
        return self.out_activation(output)