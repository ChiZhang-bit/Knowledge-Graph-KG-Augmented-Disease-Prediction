import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math


class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2] * 3
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.3)
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, x):
        device = self.device
        # input: (visit, 3, codes )
        max_len = max([(len(v[9]) + len(v[10]) + len(v[1]) + len(v[5]) + len(v[0])) for v in x])
        input_np = []
        for visit in x:
            input_tmp = []
            input_tmp.extend(visit[9])
            input_tmp.extend(list(np.array(visit[10]) + self.voc_size[0]))
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0] + self.voc_size[1]))
            input_tmp.extend(list(np.array(visit[5]) + self.voc_size[0] + self.voc_size[1] + self.voc_size[2]))
            input_tmp.extend(list(np.array(visit[0]) + self.voc_size[0] + self.voc_size[1] + self.voc_size[2] * 2))
            if len(input_tmp) < max_len:
                input_tmp.extend( [self.input_len]*(max_len - len(input_tmp)) )

            input_np.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(input_np).to(device)) # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1) # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0)) # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0)) # h: (1, visit, emb)

        g = g.squeeze(dim=0) # (visit, emb)
        h = h.squeeze(dim=0) # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1) # (visit, 1)
        attn_h = F.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        return self.output(c)