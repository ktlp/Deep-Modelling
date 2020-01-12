from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):

        self.x_data = torch.from_numpy(x_data).float()
        if y_data.dtype == float:
            self.y_data = torch.from_numpy(y_data).float()
        else:
            self.y_data = torch.from_numpy(y_data).long()

        self.len = self.x_data.size(0)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Scaler():
# add support for pytorch tensors
    def __init__(self, data):
        self.scaler = MinMaxScaler()

        # check for 1-D
        if data.ndim == 1:
            data = data.reshape(-1,1)
        self.scaler.fit(data)
        self.min = self.scaler.data_min_
        self.max = self.scaler.data_max_

    def __call__(self, data):
        if data.ndim == 1:
            data = data.reshape(-1,1)
        return self.scaler.transform(data)