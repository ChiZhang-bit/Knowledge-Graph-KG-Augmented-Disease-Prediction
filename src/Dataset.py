import torch
from torch.utils.data import TensorDataset


class DiseasePredDataset(TensorDataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # x = torch.tensor(self.data[index], dtype=torch.float32)
        # y = torch.tensor(self.targets[index], dtype=torch.float32)
        x = torch.stack([torch.squeeze(i, dim=0) for i in self.data[index]])
        y = torch.squeeze(self.targets[index], dim=0)
        return x, y
