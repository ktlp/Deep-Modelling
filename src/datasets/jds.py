from torch.utils.data import DataLoader, dataset
from src.base import BaseDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from src.datasets.main import CustomDataset, Scaler
from src.datasets.mos import MosDataset
import matplotlib.pyplot as plt
import numpy as np
class Jds_dataset(MosDataset):
    def __init__(self, root):
        super().__init__(root)
        
        df = pd.read_csv(root)
        df = df[(df['region'] == 2) | (df['region'] == 3)]
        X = df[['vgs', 'vds', 'L']].to_numpy()
        y = df['jds'].to_numpy()

        self.x_scaler, self.y_scaler = Scaler(X), Scaler(y)
        X, y = self.x_scaler(X), self.y_scaler(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        
        self.train_set = CustomDataset(X_train, y_train)
        self.test_set = CustomDataset(X_test, y_test)

