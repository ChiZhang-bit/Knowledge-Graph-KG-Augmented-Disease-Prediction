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


class DiseasePredGramDataset(TensorDataset):
    def __init__(self, features, labels, ancestor, indicator1, indicator2):
        super().__init__()
        self.feature = features
        self.label = labels
        self.ancestor = ancestor
        self.indicator1 = indicator1
        self.indicator2 = indicator2

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        # x: (seq_length, feature_size)
        # y: (dignoses size)
        x = self.feature[index]
        ancestor = self.ancestor[index]
        indicator1 = self.indicator1[index]
        indicator2 = self.indicator2[index]
        y = self.label[index]
        return x, ancestor, indicator1, indicator2, y


def load_dataset(features, ancestors, indicator1s, indicator2s, labels, split_train_point, split_test_point):
    train_features, train_labels = features[:split_train_point], labels[:split_train_point]
    train_ancestor, train_in1, train_in2 = ancestors[:split_train_point], indicator1s[:split_train_point], indicator2s[:split_train_point]

    test_features, test_labels = features[split_train_point:split_test_point], labels[split_train_point:split_test_point]
    test_ancestor, test_in1, test_in2 = ancestors[split_train_point:split_test_point], indicator1s[split_train_point:split_test_point], indicator2s[split_train_point:split_test_point]

    valid_features, valid_labels = features[split_test_point:], labels[split_test_point:]
    valid_ancestor, valid_in1, valid_in2 = ancestors[split_test_point:], indicator1s[split_test_point:], indicator2s[split_test_point:]

    train_data = DiseasePredDataset(train_features, train_labels, train_ancestor, train_in1, train_in2)
    test_data = DiseasePredDataset(test_features, test_labels, test_ancestor, test_in1, test_in2)
    valid_data = DiseasePredDataset(valid_features, valid_labels, valid_ancestor, valid_in1, valid_in2)

    return train_data, test_data, valid_data