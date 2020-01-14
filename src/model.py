import torch
import numpy as np
from src.models.Regressor import Regressor
from src.models.Classifier import Classifier
from src.trainers.regressor import Regression_Trainer
from src.trainers.classifier_trainer import Classifier_Trainer
from src.trainers.trainer import Trainer
from src.base.base_trainer import BaseTrainer
from src.base.base_dataset import BaseDataset
from src.base.base_net import BaseNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals.joblib import dump, load
class Model():
    def __init__(self, type, dataset: BaseDataset):

        assert type in ['regressor', 'classifier']

        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = r'C:\Users\Kostas\PycharmProjects\Deep-Modeling\logs\log.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.type = type
        if self.type == 'regressor':
            self.net = Regressor()
            #self.trainer = Regression_Trainer(n_epochs=100, weight_decay=1e-7)
            #self.trainer = Trainer(self.net, n_epochs=3, weight_decay=1e-7)
        else:
            self.net = Classifier()
            #self.trainer = Classifier_Trainer()

        self.trainer = Trainer(self.net, n_epochs=15, weight_decay=1e-7)

        self.dataset = dataset
        self.x_scaler = self.dataset.x_scaler
        if hasattr(self.dataset, 'y_scaler'):
            self.y_scaler = self.dataset.y_scaler
            
    def train(self):
        self.trainer.fit(self.dataset, self.net)

    def save(self, PATH):

        dump(self.x_scaler, 'x_scaler.bin', compress=True)

        save_dict = {}
        save_dict['net_dict'] = self.net.state_dict()

        if hasattr(self.dataset, 'y_scaler'):
            dump(self.y_scaler, 'y_scaler.bin', compress=True)

        torch.save(save_dict, PATH)

    def load(self, PATH):
        model_dict = torch.load(PATH)

        self.net.load_state_dict(model_dict['net_dict'])

        self.x_scaler = load('x_scaler.bin')

        if hasattr(self.dataset,'y_scaler'):
            self.y_scaler = load('y_scaler.bin')


    def __call__(self, input, scaled=False):


        # scale condition
        if not scaled:
            input_transformed = self.x_scaler.scaler.transform(input.reshape(-1,3))

        # to torch tensor
        input = torch.tensor(input_transformed).float()

        # predict
        with torch.no_grad():
            output = self.net(input).numpy()

        if (not scaled) & (hasattr(self, 'y_scaler')):
            output = self.y_scaler.scaler.inverse_transform(output)

        return output