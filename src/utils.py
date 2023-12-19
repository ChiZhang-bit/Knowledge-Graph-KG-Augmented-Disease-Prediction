import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target
import pickle
from tqdm import tqdm
import torch
from Dataset import read_data


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def get_accuracy(y, pred):
    y = np.array(y)
    pred = np.array(pred)

    y_label = np.zeros(y.shape)
    y_pred = np.zeros(pred.shape)
    y_label[y >= 0.5] = 1
    y_label[y < 0.5] = 0
    y_pred[pred >= 0.5] = 1
    y_pred[pred < 0.5] = 0

    # print(y_label.sum(axis=0))
    # print(y_label.sum(axis=0).shape)
    # 删掉那个没有疾病的维度
    y_label = np.delete(y_label, obj=32, axis=1)
    y_pred = np.delete(y_pred, obj=32, axis=1)
    
    # Micro Acc & Macro Acc
    micro_accuracy = roc_auc_score(y_label, y_pred, average="micro")
    macro_accuracy = roc_auc_score(y_label, y_pred, average="macro")

    precision_mean = precision_score(y_label, y_pred, average="micro")
    recall_mean = recall_score(y_label, y_pred, average="micro")
    f1_mean = f1_score(y_label, y_pred, average="micro")
    return micro_accuracy, macro_accuracy, precision_mean, recall_mean, f1_mean

def load_W_index(data):
    W_index_list = []
    for sample in data:
        W_index_list.append(sample['GAT_data']['W_index_list'])
    return W_index_list

#为每个visit生成固定大小的W_index和feature_index并保存到final_data里
def process_data(data, adj, neighbour_size = 512):
    #data : [4016 * list: (6*tensor(1,2850))]
        
    sample_W_index_list = []   # 将每个visit对应的图的W存进来[4016 * list:[6 * dict[128]]]
    sample_adj_index_list = []  # 将每个visit对应的图的neighbour存进来[4016 * list:[6 * list[128]]]
    neighbour_num_list = []  # 存储的是每个visit中本来有的neighbour个数[4016 * list:[6]]
    for sample in tqdm(data, total = len(data), desc = "generating neighbour_index"):
        num_list = []   #一个sample中每个visit的neighbour个数,len(num_list) = 6
        adj_index_list = [] #len(adj_index_list) = 6
        w_index_list = []   #len(w_index_list) = 6
        for visit in sample:
            visit_index = [i for i in range(visit.shape[1]) if visit[0][i] != 0]
            neighbour_num = 0
            w_index = {}    #len(w_index) = 128
            adj_index = []  #len(adj_index) = 128
            for i in visit_index:
                for j in range(adj.shape[1]):
                    if adj[i,j] != 0:
                        neighbour_num += 1
                        adj_index.append(j)
                        w_index[(i, j)] = adj[i,j] - 1
                    if len(adj_index) >= neighbour_size:#边的个数已经达到上限
                        break
                if len(adj_index) >= neighbour_size:#边的个数已经达到上限
                    break
            if len(adj_index) < neighbour_size:    #若边的个数不够则补充
                for k in range(neighbour_size - len(adj_index)):
                    adj_index.append(2850 + k)
                    w_index[(2850 + k, 2850 + k)] = 11
            assert len(adj_index) == neighbour_size
            num_list.append(neighbour_num)
            w_index_list.append(w_index)
            adj_index_list.append(adj_index)
        neighbour_num_list.append(num_list)
        sample_W_index_list.append(w_index_list)
        sample_adj_index_list.append(adj_index_list)
            
    print(f'sample_W_index_list:{len(sample_W_index_list)}')    #4016
    print(f'sample_W_index_list[0]:{len(sample_W_index_list[0])}')  #6
    print(f'sample_W_index_list[0][0]:{len(sample_W_index_list[0][0])}') #128
    print(f'sample_adj_index_list:{len(sample_adj_index_list)}')    #4016
    print(f'sample_adj_index_list[0]:{len(sample_adj_index_list[0])}')  #6
    print(f'sample_adj_index_list[0][0]:{len(sample_adj_index_list[0][0])}') #128
    print(f'neighbour_num_list:{len(neighbour_num_list)}')    #4016
    print(f'neighbour_num_list[0]:{len(neighbour_num_list[0])}')  #6
    with open(f'../data/sample_W_index_list_{neighbour_size}.pkl', 'wb') as f:
        pickle.dump(sample_W_index_list, f)
    with open(f'../data/sample_adj_index_list_{neighbour_size}.pkl', 'wb') as f:
        pickle.dump(sample_adj_index_list, f)
    with open(f'../data/neighbour_num_list_{neighbour_size}.pkl', 'wb') as f:
        pickle.dump(neighbour_num_list, f)
    
    
    #找到每一个visit 把one-hot转换成index list
    # with open('../data/sample_W_index_list.pkl', 'rb') as f:
    #     sample_W_index_list = pickle.load(f)
    sample_visit_index = []  #[4016 * list:[6 * list:[128]]]
    for sample in tqdm(sample_W_index_list, total = len(sample_W_index_list), desc = "generating feature_index"):  #sample[6 * dict(128)]
        visit_index = []    #list:[6 * list:[128]]
        for visit in sample: #visit dict(128)
            feature_index = [key[0] for key, value in visit.items()] #list:[128]
            visit_index.append(feature_index)
        sample_visit_index.append(visit_index)
    print(f'sample_visit_index:{len(sample_visit_index)}')  #4016
    print(f'sample_visit_index[0]:{len(sample_visit_index[0])}')    #6
    print(f'sample_visit_index[0][0]:{len(sample_visit_index[0][0])}')    #128
    with open(f'../data/sample_visit_index_{neighbour_size}.pkl', 'wb') as f:
        pickle.dump(sample_visit_index, f)

    # visit_graph: (useful_visit_size, feature_index)
    # visit_graph:
    # [ torch.tensor([2,3,4]), torch.tensor([5,6,7,8]), ... ]
    
    # with open('../data/')
    # with open('../data/sample_visit_index.pkl', 'rb') as f:
    #     sample_visit_index = pickle.load(f) #[4016 * [6 * [128]]]
    # with open('../data/sample_adj_index_list.pkl', 'rb') as f:
    #     sample_adj_index = pickle.load(f)   #[4016 * [6 * [128]]]
    # with open('../data/sample_W_index_list.pkl', 'rb') as f:
    #     sample_W_index_list = pickle.load(f)    #[4016 * [6 * {128}]]
    # with open('../data/neighbour_num_list.pkl', 'rb') as f:
    #     neighbour_num_list = pickle.load(f)     #[4016 * [6]]
    final_data = []
    for i in tqdm(range(len(sample_visit_index)), total = len(sample_visit_index), desc = "generating final_data"):
        sample_data = {
            'left_data': data[i],   #[6 * tensor(1,2850)]
            'GAT_data':{
                'visit_index': [torch.tensor(fs).unsqueeze(dim = 0) for fs in sample_visit_index[i]],   #[6 * tensor(1,128)]
                'adj_index': [torch.tensor(fs).unsqueeze(dim = 0) for fs in sample_adj_index_list[i]],   #[6 * tensor(1,128)]
                'W_index_list': sample_W_index_list[i], #[6 * {128}]
                'num_list': neighbour_num_list[i]   #[6]
            }
        }
        final_data.append(sample_data)
    with open(f'../data/final_data_{neighbour_size}.pkl', 'wb') as f:
        pickle.dump(final_data, f)
    print(f'final_data:{len(final_data)}')#4016
    print(f'final_data[0]["left_data"]:',final_data[0]['left_data'])

def generate_gram_data():
    with open('data/code_map_10.pkl', 'rb') as f:
        code_map_10 = pickle.load(f)
    code_map_10_keys = list(code_map_10.keys())
    code_map_10_keys = code_map_10_keys[:1296]
    add_keys = []
    for key in code_map_10_keys:
        if type(key) != str:
            continue
        if len(key) == 4 and key[:3] not in code_map_10_keys and key[:3] not in add_keys:
            add_keys.append(key[:3])
        elif len(key) == 5:
            if key[:4] not in code_map_10_keys and key[:4] not in add_keys:
                add_keys.append(key[:4])
            if key[:3] not in code_map_10_keys and key[:3] not in add_keys:
                add_keys.append(key[:3])
    code_map_10_keys = code_map_10_keys + add_keys
    print(len(code_map_10_keys))

    features_one_hot = torch.load('data/features_one_hot.pt')   #[5995 * [6 * tensor(1,2850)]]
    feature_ancestor_information = []
    for sample in tqdm(features_one_hot, total = len(features_one_hot), desc = "feature_ancestor_information"):
        sample_ancestor_information = []
        for visit in sample: #visit:tensor(1,2850)
            visit_icd_list = []
            for index in range(1296):
                if visit[0][index] != 0:
                    visit_icd_list.append(code_map_10_keys[index])
            for code in visit_icd_list: #添加每个code对应的高层code到visit_icd_list中
                if type(key) != str:
                    continue
                if len(code) == 4 and code[:3] not in visit_icd_list:
                    visit_icd_list.append(code[:3])
                elif len(code) == 5:
                    if code[:4] not in visit_icd_list:
                        visit_icd_list.append(code[:4])
                    if code[:3] not in visit_icd_list:
                        visit_icd_list.append(code[:3])
            visit_ancestor_information = {}
            for i in range(len(visit_icd_list)):
                for j in range(len(visit_icd_list)):
                    if i != j and visit_icd_list[i].startswith(visit_icd_list[j]):
                        visit_ancestor_information[i] = j
            sample_ancestor_information.append(visit_ancestor_information)
        feature_ancestor_information.append(sample_ancestor_information)
    #feature_ancestor_information: [5995 * [6 * {}]]
    print(len(feature_ancestor_information))
    print(len(feature_ancestor_information[0]))
    with open('ancestor_information.pkl', 'rb') as f:
        ancestor_information = pickle.load(f)
    print(ancestor_information[0][1])
    
# generate_gram_data()