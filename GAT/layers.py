import numpy as np
import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    处理图的输入特征嵌入
    """
    def __init__(self, nfeat, nemb):
        """
        nfeat: 输入特征的数量
        nemb: 嵌入向量的维度
        """
        super().__init__()
        self.embedding = nn.Embedding(nfeat, nemb)  # 将输入特征映射到nemb维度  x: (*, nembedding_dim)
        # 只要把他的邻居和他的embedding筛出来就行了
        nn.init.xavier_uniform_(self.embedding.weight)
        print ("torch.min(torch.abs(self.embedding.weight)): ", torch.min(torch.abs(self.embedding.weight)))
        print ("torch.max(torch.abs(self.embedding.weight)): ", torch.max(torch.abs(self.embedding.weight)))

    def forward(self, x):
        """
        :param x:   (feature_index_num)
        :return:    embeddings B*F*E
        """

        # 这里只有一个2850*embedding_size的矩阵 (F*E)
        emb = self.embedding(x)   # feature_index_num * embedding_size
        
        return emb   # F*E

def get_triu_indices(n, diag_offset=1):
    """
    返回上三角部分的行列索引，diag_offset=1则包含对角线上的元素
    get the row, col indices for the upper-triangle of an (n, n) array
    """
    return np.triu_indices(n, diag_offset)

def get_all_indices(n):
    """
    返回方阵中所有元素的行列索引
    get all the row, col indices for an (n, n) array
    """
    return map(list, zip(*[(i, j) for i in range(n) for j in range(n)]))