import pandas as pd
from sklearn.model_selection import train_test_split
from src.datasets.main import CustomDataset, Scaler
from src.datasets.mos import MosDataset



class Jds_dataset(MosDataset):
    def __init__(self, root, label = 'jds'):
        super().__init__(root)

        df = pd.read_csv(root)
        df = df[(df['region'] == 2) | (df['region'] == 3)]
        X = df[['vgs', 'vds', 'L']].to_numpy()
        y = df[label].to_numpy()

        self.x_scaler, self.y_scaler = Scaler(X), Scaler(y)
        X, y = self.x_scaler(X), self.y_scaler(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        self.train_set = CustomDataset(X_train, y_train)
        self.test_set = CustomDataset(X_test, y_test)


class Region_dataset(MosDataset):
    def __init__(self, root):
        super().__init__(root)

        df = pd.read_csv(root)
        X= df[['vgs','vds','L']].to_numpy()
        y = df['region'].to_numpy().astype(int)
        self.x_scaler = Scaler(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        self.test_set = CustomDataset(X_test, y_test)
        self.train_set = CustomDataset(X_train, y_train)

class Vdsat_dataset(MosDataset):
    def __init__(self, root):
        super().__init__(root)

        df = pd.read_csv(root)
        X= df[['vgs','vds','L']].to_numpy()
        y = df['vdsat'].to_numpy()
        self.x_scaler = Scaler(X)
        self.y_scaler = Scaler(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        self.test_set = CustomDataset(X_test, y_test)
        self.train_set = CustomDataset(X_train, y_train)

class Vth_dataset(MosDataset):
    def __init__(self, root):
        super().__init__(root)

        df = pd.read_csv(root)
        X= df[['vgs','vds','L']].to_numpy()
        y = df['vth'].to_numpy()
        self.x_scaler = Scaler(X)
        self.y_scaler = Scaler(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        self.test_set = CustomDataset(X_test, y_test)
        self.train_set = CustomDataset(X_train, y_train)

