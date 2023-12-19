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


class GRAM(nn.Module):
    def __init__(self, input_dim, num_ancestors, embed_dim, hidden_dim, output_dim, attn_dim):
        super(GRAM, self).__init__()
        
        self.input_dim = input_dim
        self.num_ancestors = num_ancestors
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.attn_dim = attn_dim
        
        self.embedding = nn.Embedding(2850 + num_ancestors, 
                                      embed_dim,
                                      padding_idx=input_dim + num_ancestors)
        
        self.attention_mlp = AttentionMLP(
            self.embed_dim, self.attn_dim
        )
        
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_dim, output_dim)
        self.out_activation = nn.Sigmoid()

    def forward(self, x, ancestors, indicator1, indicator2):
        # x: (batch_size, visit_num, input_size(256))
        # leaves: (batch_size, visit_num, leaves_size 256)
        # ancestors: (batch_size, visit_num, ancestor_size 256)
        # indicator: mask leaves & mask ancestors (batch_size, visit_num, 256[1/0])
        emb_x = self.embedding(x)  # (batch_size, visit_num, input_size, embed_size)
        emb_ancestors = self.embedding(ancestors)
        # mask 掉多余的信心
        emb_x = torch.einsum("bvie,bvi -> bvie", emb_x, indicator1)
        emb_ancestors = torch.einsum("bvie,bvi -> bvie", emb_x, indicator2)
        # (batch_size, visit_num, 256, embed_size)
        
        # 注意力机制：
        attention_weights = torch.zeros(emb_ancestors.shape[:-1], device=emb_ancestors.device)
        # (batch_size, visit_num, 256)

        for i in range(emb_x.shape[2]):
            # 对于每个叶子节点和祖先节点的嵌入计算注意力机制
            attn_scores = self.attention_mlp(emb_x[:, :, i, :], emb_ancestors[:, :, i, :])
            attention_weights[:, :, i] = attn_scores  # (batch_size, visit_num)

        attention_weights = F.softmax(attention_weights, dim=-1)
        # (batch_size, visit_num, 256)

        # 使用注意力权重加权求和
        emb_weighted = torch.einsum('bvla, bvl->bva', emb_x, attention_weights)
        emb_weighted = F.dropout(emb_weighted, 0.1)
        # (batch_size, visit_num, embed_size)

        rnn_out, _ = self.rnn(emb_weighted)
        # rnn_out: (batch_size, visit_num, hidden_size)

        output = self.linear(rnn_out)
        # output: (batch_size, visit_num, output_dim)

        output = torch.sum(output, dim=1)  # batch_size, output_dim
        return self.out_activation(output)
        
        

        