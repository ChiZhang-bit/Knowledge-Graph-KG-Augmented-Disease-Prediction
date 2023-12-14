import torch
from torch import nn
import torch.nn.functional as F
from layers import get_all_indices, Embedding


class GraphAttentionLayer(nn.Module):

    def __init__(self, ninfeat, noutfeat, dropout, alpha, nrelation=11):
        """
        ninfeat: 输入特征的维度
        noutfeat: 输出特征的维度
        """
        super().__init__()

        self.W = nn.ParameterList()
        
        for _ in range(nrelation):
            # 根据关系确定不同的W矩阵 
            # W[0] - W[10]表示11种不同的关系
            self.W.append(nn.Parameter(torch.zeros(size=(ninfeat, noutfeat))))
            nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
        
        self.linear = nn.Linear(2 * noutfeat, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, n_index, w_index: dict):
        '''
        :param x:       feature_num, embed_size
        :param adj:     邻接矩阵
        :param n_index: 子图的节点集合
        :param w_index: dict 采用的权重 (i,j) -> w_index 
        :param feature_num: int 表示本身feature的个数 (不含neighbour)
        :return:        FloatTensor B*F*(headxE2)
        '''
        # x:  feature_num, embed_size
        n_index = n_index.tolist()
        h_list = []        
        # 根据边的类别不同选择不同的W矩阵来运算
        e_list = []
        neighbour_temp_list = [] #  (relation_nums, 1)
        for key, value in w_index.items():
            neighbour_embed = x[n_index.index(key[1]), :] # (embedding_size)  
            # W: (ninfeat==embedding_size, noutfeat)
            print(neighbour_embed.shape)
            print(self.W[value].shape)
            neighbour_temp = torch.einsum("i,ij->ij", neighbour_embed, self.W[value])  # (embedding_size, noutfeat) 
            print("neighbour_temp.shape:", neighbour_temp.shape)
            neighbour_temp_list.append(neighbour_temp)  

            feature_embed = x[n_index.index(key[0]), :]   # (embdedding_size)
            feature_temp = torch.einsum("i,ij->ij", feature_embed, self.W[value])  # (embedding_size, noutfeat) 
            # feature_temp = torch.matmul(feature_embed, self.W[value])
            
            # print(f'neighbour_temp:{neighbour_temp.shape},feature_temp:{feature_temp.shape}')
            
            # 拼接:
            hh = torch.cat([feature_temp, neighbour_temp], dim=-1)  # (embedding_size, noutfeat*2)
            print(f'hh:{hh.shape}')
            e = self.leakyrelu(self.linear(hh))  # (embedding_size, 1)
            print(f'e:{e.shape}')
            e_list.append(e)
        
        attn = torch.cat(e_list, dim = 1) #  (embedding_size, relation_nums)
        attn = self.dropout(F.softmax(attn, dim=1)) #  (embedding_size, relation_nums)
        for i, neighbour_temp in enumerate(neighbour_temp_list):
            print(attn[:, i].shape)  # (embedding_size)
            print(neighbour_temp.shape) # (embedding_size, noutfeat) 

            h_list.append(torch.einsum('i,ij->ij', attn[:, i], neighbour_temp))  # (1, noutfeat)
        
        hi = torch.cat(h_list, dim=1)  # (n, foutfeat)
        return torch.sum(hi, dim=0)  # (foutfeat)


class GATModel(nn.Module):

    def __init__(self, nfeat, nemb, gat_layers, gat_hid, dropout, alpha=0.2, nrelation=11):
        """
        nfeat: feature_size
        nemb: 每个特征的嵌入维度
        gat_layers: 图注意力层的数量
        gat_hid: 图注意力的隐藏单元数
        """
        super().__init__()

        self.embedding = Embedding(nfeat, nemb)
        self.gat_layers = gat_layers
        self.gats = torch.nn.ModuleList()
        self.nrelation = nrelation
        ninfeat = nemb
        for _ in range(gat_layers):
            self.gats.append(
                GraphAttentionLayer(
                    ninfeat=ninfeat, 
                    noutfeat=gat_hid,
                    dropout=dropout, 
                    alpha=alpha,
                    nrelation=self.nrelation
                )
            )
            # 第一层是embedding后的输出作为输入，之后是GATlayer的输出作为输入：
            ninfeat = gat_hid  
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj=None):
        """
        not supported for batch process
        :param x:       (visit_size, feature_size)
        :param adj:     FloatTensor F*F, default fully connected
        """
        # 1. 找到每一个visit 把one-hot转换成index list
        x_list = x.tolist()
        visit_graph = []  # visit_graph表示n次visit涉及到的feature节点的index
        W_index_list = []   # 将每个visit对应的图存进来
        feature_num_list = []  # 存储的是每个visit中本来有的feature种类个数
        for visit in x_list:
            # visit: (feature_size)
            feature_index = [i for i, value in enumerate(visit) if value != 0]
            if len(feature_index) == 0:  # 若visit全为0就筛掉
                continue
            feature_num_list.append(len(feature_index))
            # 根据临界矩阵扩展：
            adj_index = []
            w_index = {}
            for i in feature_index:
                for j in range(adj.shape[0]):
                    if adj[i,j]!= 0:
                        adj_index.append(j)
                        w_index[(i, j)] = adj[i,j] - 1
            W_index_list.append(w_index)
            feature_index.extend(adj_index)
            visit_graph.append(torch.tensor(feature_index))
        # visit_graph: (useful_visit_size, feature_index)
        # visit_graph:
        # [ torch.tensor([2,3,4]), torch.tensor([5,6,7,8]), ... ]

        # 2. embedding feature
        visit_h = []    # visit_h: (visit_size, feature_num(不相同), embedding_size)
        for visit in visit_graph:
            visit_h.append(self.embedding(visit))  # (append进去的: feature_num, embed_size)
        
        # 3. Graph Attention Layers
        h_subgraph = []  # 所有visit对应的子图
        for l in range(self.gat_layers):
            print(f"----------------No.{l}.gat_layers-----------------------")
            for i, visit in enumerate(visit_h):
                # 这里的visit_h[i] 是第i个子图的embedding: (feature_num ,embed_size)
                h_v = self.gats[l](x = visit_h[i],
                                   n_index = visit_graph[i], 
                                   w_index = W_index_list[i])  # 这里获得一个子图 # (noutfeat)
                h_subgraph.append(F.elu(self.dropout(h_v)).unsqueeze(0))  # (1, noutfeat)
        
        h_subgraph_embed = torch.cat(h_subgraph, dim=0) # (visit_num, noutfeat)
        h = torch.mean(h_subgraph_embed, dim=0)  # (noutfeat)
        return h
        

# batch_size = 2
# feature_size = 5
    
# input: (visit_size, feature_size)

model = GATModel(
    nfeat=5,
    nemb=128,
    gat_layers=1,
    gat_hid=16, #  == noutfeat
    dropout=0.1
)

x = torch.tensor(
    [
        [1,0,0,1,0],
        [1,0,0,0,0],
    ],
)

adj = torch.tensor(
    [
        [1,0,0,1,0],
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,1]
    ]
)

# 1. 找到每一个visit 把one-hot转换成index list
# 3. 根据index list 找到 neighbor
# 2. x =[ tensor([1, 4, 5]), tensor([2, 4, 6, 8]) ]

print(model(x, adj))

# [1*10, 0*others]
# 假设10个中心节点，55个

