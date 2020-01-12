from torch.utils.data import DataLoader, dataset
from src.base import BaseDataset
import pandas as pd
from src.datasets.main import CustomDataset
from sklearn.model_selection import train_test_split

class MosDataset(BaseDataset):
    def __init__(self, root):
        super().__init__(root)
        self.X_vector = ['vgs', 'vds', 'vbs', 'L']
        self.X_vector_advanced = ['vgs', 'vds', 'vbs', 'L', 'W']


    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
                DataLoader, DataLoader):

        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader
