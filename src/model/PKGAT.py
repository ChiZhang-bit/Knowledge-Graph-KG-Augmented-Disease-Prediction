import torch
from torch import nn
import torch.nn.functional as F


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
        self.embedding = nn.Embedding(nfeat, nemb)  # 将输入特征映射到nemb维度  x: (*, nembedding_dim) 每轮epoch参数是否有变化？
        # 只要把他的邻居和他的embedding筛出来就行了
        nn.init.xavier_uniform_(self.embedding.weight)
        # print ("torch.min(torch.abs(self.embedding.weight)): ", torch.min(torch.abs(self.embedding.weight)))
        # print ("torch.max(torch.abs(self.embedding.weight)): ", torch.max(torch.abs(self.embedding.weight)))

    def forward(self, x):
        """
        :param x:   (feature_index_num)
        :return:    embeddings B*F*E
        """

        # 这里只有一个2850*embedding_size的矩阵 (F*E)
        emb = self.embedding(x)   # feature_index_num * embedding_size
        
        return emb   # F*E


class GraphAttentionLayer(nn.Module):

    def __init__(self, ninfeat, noutfeat, dropout, alpha, nrelation=11, device=torch.device("cuda")):
        """
        ninfeat: 输入特征的维度
        noutfeat: 输出特征的维度
        """
        super().__init__()

        self.W = nn.ParameterList()
        
        for _ in range(nrelation + 1):
            # 根据关系确定不同的W矩阵 
            # W[0] - W[10]表示11种不同的关系 第12种表示虚关系
            self.W.append(nn.Parameter(torch.zeros(size=(ninfeat, noutfeat))))
            # if _ is not nrelation:  # 虚关系的W默认是0
            nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
        
        self.linear = nn.Linear(2 * noutfeat, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.device = device

    def forward(self, feature_embed, neighbour_embed, w_index, indicator):
        '''
        :param feature_embed:   tensor(batch_size * 6 * neighbour_size * embedding_size)
        :param neighbour_embed: tensor(batch_size * 6 * neighbour_size * embedding_size)
        :param w_index: [batch_size * [6 * {neighbour_size}]]
        :param indicator: tensor(batch_size, 6 , neighbour_size) 每次运算时mask掉的01 tensor
        :return:        FloatTensor B*F*(headxE2)
        '''
        #拼W
        W_matrix = []
        # print(self.W[0])
        # W = self.W
        # W = W.to('cpu')
        # print('W[0]:',W[0])
        for batch in w_index:   # batch[6 * {neighbour_size}]
            W_matrix_batch = []
            for visit in batch: # visit{neighbour_size} neighbour_size个关系
                W = self.W.to('cpu')
                W_matrix_visit = [] # [neighbour_size * tensor(ninfeat * noutfeat)]
                for v in visit.values():
                    if int(v) < len(W):
                        W_matrix_visit.append(W[int(v)])
                    else:
                        print(v)
                        input()
                W_matrix_visit = torch.stack(W_matrix_visit, dim = 0) # tensor(neighbour_size * ninfeat * noutfeat)
                W_matrix_batch.append(W_matrix_visit)
            W_matrix_batch = torch.stack(W_matrix_batch, dim = 0) #tensor(6 * neighbour_size * ninfeat * noutfeat)
            W_matrix.append(W_matrix_batch)
        W_matrix = torch.stack(W_matrix, dim = 0) #tensor(batch_size * 6 * neighbour_size * ninfeat * noutfeat)
        W_matrix = W_matrix.cuda()

        # GAT Core:

        # W_matrix: 
        neighbour = torch.einsum('bvne, bvnef -> bvnf', neighbour_embed,W_matrix)    #tensor(batch_size * 6 * neighbour_size * noutfeat)
        feature = torch.einsum('bvne, bvnef -> bvnf', feature_embed,W_matrix)    #tensor(batch_size * 6 * neighbour_size * noutfeat)
        hh = torch.cat([neighbour, feature], dim = -1)  #tensor(batch_size * 6 * neighbour_size * (2 * noutfeat))
        e = self.leakyrelu(self.linear(hh)) #tensor(batch_size * 6 * neighbour_size * 1)
        attn = self.dropout(F.softmax(e, dim = -2)) #tensor(batch_size * 6 * neighbour_size * 1)

        # mask掉虚边生成attn:
        indicator = indicator.unsqueeze(dim=-1)  # tensor(batch_size * 6 * neighbour_size * 1)
        attn =  torch.mul(indicator, attn)

        h_list = torch.einsum("bvej,bveo->bvo", attn, neighbour) # (batch_size, 6, outfeat)
        return h_list

class GATModel(nn.Module):

    def __init__(self, nfeat, nemb, gat_layers, gat_hid, dropout, alpha=0.2, nrelation=11, device=torch.device("cuda")):
        """
        nfeat: feature_size
        nemb: 每个特征的嵌入维度
        gat_layers: 图注意力层的数量
        gat_hid: 图注意力的隐藏单元数
        """
        super().__init__()

        self.embedding = Embedding(nfeat+256, nemb) # 增加虚节点的embedding
        self.gat_layers = gat_layers
        self.gats = torch.nn.ModuleList()
        self.nrelation = nrelation
        self.device = device
        ninfeat = nemb
        for _ in range(gat_layers):
            self.gats.append(
                GraphAttentionLayer(
                    ninfeat=ninfeat, 
                    noutfeat=gat_hid,
                    dropout=dropout, 
                    alpha=alpha,
                    nrelation=self.nrelation,
                    device=self.device
                )
            )
            # 第一层是embedding后的输出作为输入，之后是GATlayer的输出作为输入：
            ninfeat = gat_hid  
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feature_index, neighbour_index, W_index, indicator):
        """
        feature_index : , #tensor(batch_size * 6 * 128)
        neighbour_index: ,    #tensor(batch_size * 6 * 128)
        batch_W_index: ,  #[batch_size * [6 * {128}]]
        indicator: tensor(batch_size, 6 , 128) [1, ... , 0, ...]
        """

        # 1. embedding feature
        feature_embed = self.embedding(feature_index)  # tensor(batch_size * 6 * 128 * embedding_size)
        neighbour_embed = self.embedding(neighbour_index)  # tensor(batch_size * 6 * 128 * embedding_size)

        # mask 掉不存在的边和点
        # feature_embed = feature_embed * indicator
        # neighbour_embed = neighbour_embed * indicator

        # print('feature_embed.shape:',feature_embed.shape)
        # print('neighbour_embed.shape:',neighbour_embed.shape)
        
        # 2. Graph Attention Layers
        for l in range(self.gat_layers):
            # print(f"----------------No.{l}.gat_layers-----------------------")
            h_v = self.gats[l](feature_embed = feature_embed,
                                neighbour_embed = neighbour_embed, 
                                w_index = W_index,
                                indicator = indicator)  # (batch_size, 6, outfeat)
            
            h_v = torch.mean(h_v, dim = 1)  #h_v: (batch_size, outfeat)
                
        return h_v