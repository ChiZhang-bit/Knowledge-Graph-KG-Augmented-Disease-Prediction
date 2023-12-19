import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import pickle


def del_tensor_ele(arr: torch.tensor, index):
    """
    删除指定位置的一维tensor元素
    """
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


def read_data(feature_file, label_file, device):
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)
    # features = torch.load(feature_file)
    labels = torch.load(label_file) # every element in list : [1, 100]
    
    sum_tensor = torch.sum(torch.stack(labels).squeeze(dim=1), dim=0).to(device)
    
    nums_disease = {i:num for i, num in enumerate(sum_tensor.tolist())}
    nums_disease = dict(sorted(nums_disease.items(), key=lambda item: item[1]))
    delete_index = list(nums_disease.keys())[0:10]
    
    new_labels = []
    for i in tqdm(labels, total=len(labels)):
        i.to(device)
        label = torch.squeeze(i, dim=0)
        for idx in delete_index:
            label = del_tensor_ele(label, idx)
        new_labels.append(label)
    
    # print(len(features), len(new_labels))
    # assert len(features) == len(new_labels)
    return features, new_labels


class DiseasePredDataset(TensorDataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # x: (seq_length, feature_size)
        # y: (dignoses size)
        x = self.data[index]
        x1 = torch.cat(x['left_data'], dim = 0)   #tensor(6, 2850)
        visit_index = torch.cat(x['GAT_data']['visit_index'], dim = 0)   #tensor(6, neighbour_size)
        adj_index = torch.cat(x['GAT_data']['adj_index'], dim = 0)   #tensor(6, neighbour_size)
        
        num_list = x['GAT_data']['num_list'] # list [6]
        
        indicator = torch.zeros(6, 256)
        for i, num in enumerate(num_list):
            # 默认边的个数的最大值为neighbour_size
            indicator[i, :num] = 1
        # indicator： tensor (6, neighbour_size)

        y = torch.squeeze(self.targets[index], dim=0)
        return x1, visit_index, adj_index, indicator, y