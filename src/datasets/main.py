from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

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
    def __init__(self, data, exp=None):
        self.scaler = MinMaxScaler()
        self.exp = exp

        # check for 1-D
        if data.ndim == 1:
            data = data.reshape(-1,1)
        if self.exp:
            self.scaler.fit(np.power(10,data))
        else:
            self.scaler.fit(data)
        self.min = self.scaler.data_min_
        self.max = self.scaler.data_max_

    def inverse_transform(self, data):
        data = self.scaler.inverse_transform(data.reshape(-1,1))
        if self.exp:
            data = np.log10(data)
        return data.reshape(-1,)

    def __call__(self, data):
        if data.ndim == 1:
            data = data.reshape(-1,1)
        if self.exp:
            return self.scaler.transform(np.power(10, data))
        else:
            return self.scaler.transform(data)