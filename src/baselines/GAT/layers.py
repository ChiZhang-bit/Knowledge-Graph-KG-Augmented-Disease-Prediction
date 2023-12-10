import numpy as np
import torch
import torch.nn as nn

class UciEmbedding(nn.Module):
    ''' simplified version for ALL numerial features '''

    def __init__(self, nfeat, nemb):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(nfeat, nemb)) # F*E
        nn.init.xavier_uniform_(self.embedding)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F
        :return:    embeddings B*F*E
        """
        return torch.einsum('bf,fe->bfe', x, self.embedding)    # B*F*E

class Embedding(nn.Module):

    def __init__(self, nfeat, nemb):
        super().__init__()
        self.embedding = nn.Embedding(nfeat, nemb)
        nn.init.xavier_uniform_(self.embedding.weight)
        print ("torch.min(torch.abs(self.embedding.weight)): ", torch.min(torch.abs(self.embedding.weight)))
        print ("torch.max(torch.abs(self.embedding.weight)): ", torch.max(torch.abs(self.embedding.weight)))

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    embeddings B*F*E
        """
        # print ("x: ", x)
        emb = self.embedding(x['ids'])                          # B*F*E
        # print ("emb shape: ", emb.shape)
        # print ("emb: ", emb)
        # print ("(emb * x['vals'].unsqueeze(2)) shape: ", (emb * x['vals'].unsqueeze(2)).shape)
        # print ("emb * x['vals'].unsqueeze(2): ", emb * x['vals'].unsqueeze(2))
        return emb * x['vals'].unsqueeze(2)                     # B*F*E

class Linear(nn.Module):

    def __init__(self, nfeat):
        super().__init__()
        self.weight = nn.Embedding(nfeat, 1)
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    linear transform of x
        """
        linear = self.weight(x['ids']).squeeze(2) * x['vals']   # B*F
        return torch.sum(linear, dim=1) + self.bias             # B

class FactorizationMachine(nn.Module):

    def __init__(self, reduce_dim=True):
        super().__init__()
        self.reduce_dim = reduce_dim

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        """
        square_of_sum = torch.sum(x, dim=1)**2                  # B*E
        sum_of_square = torch.sum(x**2, dim=1)                  # B*E
        fm = square_of_sum - sum_of_square                      # B*E
        if self.reduce_dim:
            fm = torch.sum(fm, dim=1)                           # B
        return 0.5 * fm                                         # B*E/B

def get_triu_indices(n, diag_offset=1):
    """get the row, col indices for the upper-triangle of an (n, n) array"""
    return np.triu_indices(n, diag_offset)

def get_all_indices(n):
    """get all the row, col indices for an (n, n) array"""
    return map(list, zip(*[(i, j) for i in range(n) for j in range(n)]))

class MLP(nn.Module):

    def __init__(self, ninput, nlayers, nhid, dropout, noutput=1):
        super().__init__()
        layers = list()
        for i in range(nlayers):
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            ninput = nhid
        if nlayers==0: nhid = ninput
        layers.append(nn.Linear(nhid, noutput))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)

def normalize_adj(adj):
    """normalize and return a adjacency matrix (numpy array)"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)