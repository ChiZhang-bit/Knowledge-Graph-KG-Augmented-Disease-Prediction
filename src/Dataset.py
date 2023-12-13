import torch
from torch.utils.data import TensorDataset

def del_tensor_ele(arr: torch.tensor, index):
    """
    删除指定位置的一维tensor元素
    """
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


def read_data(feature_file, label_file):
    features = torch.load(feature_file)
    labels = torch.load(label_file) # every element in list : [1, 100]
    sum_tensor = torch.sum(torch.stack(labels).squeeze(dim=1), dim=0)    
    
    nums_disease = {i:num for i, num in enumerate(sum_tensor.tolist())}
    nums_disease = dict(sorted(nums_disease.items(), key=lambda item: item[1]))
    delete_index = list(nums_disease.keys())[0:10]
    
    new_labels = []
    for i in labels:
        label = torch.squeeze(i, dim=0)
        for idx in delete_index:
            label = del_tensor_ele(label, idx)
        new_labels.append(label)
    
    assert len(features) == len(labels)
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
        x = torch.stack([torch.squeeze(i, dim=0) for i in self.data[index]])
        y = torch.squeeze(self.targets[index], dim=0)
        return x, y
