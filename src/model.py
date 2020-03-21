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
import os
from joblib import dump, load
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
        else:
            self.net = Classifier()

        self.trainer = Trainer(self.net, n_epochs=100, weight_decay=1e-7, early_stopping=True)

        self.dataset = dataset
        self.x_scaler = self.dataset.x_scaler
        if hasattr(self.dataset, 'y_scaler'):
            self.y_scaler = self.dataset.y_scaler
            
    def train(self):
        self.trainer.fit(self.dataset, self.net, validate=1)

    def test(self):
        self.trainer.score(self.dataset, self.net)

    def train_validate(self):
        self.trainer
    def save(self, PATH,name):

        PATH = os.path.join(PATH, name)
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        else:
            print('Overwritting model files..')

        dump(self.x_scaler, os.path.join(PATH,'x_scaler.bin'), compress=True)

        save_dict = {}
        save_dict['net_dict'] = self.net.state_dict()

        if hasattr(self.dataset, 'y_scaler'):
            dump(self.y_scaler, os.path.join(PATH,'y_scaler.bin'), compress=True)

        torch.save(save_dict, os.path.join(PATH,'model'))

    def load(self, PATH, name):

        PATH = os.path.join(PATH, name)

        model_dict = torch.load(os.path.join(PATH,'model'))

        self.net.load_state_dict(model_dict['net_dict'])

        self.x_scaler = load(os.path.join(PATH,'x_scaler.bin'))

        if hasattr(self.dataset,'y_scaler'):
            self.y_scaler = load(os.path.join(PATH,'y_scaler.bin'))


    def __call__(self, input, scaled=False):


        # scale condition
        if not scaled:
            if input.shape[1] != 3:
                input = np.transpose(input)
            input_transformed = self.x_scaler.scaler.transform(input)

        # to torch tensor
        input = torch.tensor(input_transformed).float()

        # predict
        with torch.no_grad():
            output = self.net(input).numpy()

        if (not scaled) & (hasattr(self, 'y_scaler')):
            output = self.y_scaler.scaler.inverse_transform(output)
        if self.type == 'classifier':
            output = np.argmax(output,axis=1)
        return output

class Robust_model():
    def __init__(self, path, name, model_name):
        self.path = path    # models path
        self.name = name    # robust_nmos_region
        self.model_name   = model_name  # region, jds_lower
        self.model = None
        self.load()

    def save(self):
        _ = dump(self.model, os.path.join(self.path,self.name, self.model_name))
        _ = dump(self.scaler, os.path.join(self.path,self.name, 'scaler.joblib'))

    def load(self):

        PATH = os.path.join(self.path,self.name, self.model_name)
        SCALER_PATH = os.path.join(self.path,self.name, 'scaler.joblib')

        self.model = load(PATH)
        self.scaler = load(SCALER_PATH)


    def __call__(self, input):
        input_scaled = self.scaler.transform(input)
        output = self.model.predict(input_scaled)
        return output
