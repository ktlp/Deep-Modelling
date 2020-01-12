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

from src.datasets.jds import Jds_dataset
from src.datasets.region import Region_dataset
a = Jds_dataset(r'C:\Users\Kostas\PycharmProjects\Deep-Modeling\data\nmos_40nm.csv')
b =  Region_dataset(r'C:\Users\Kostas\PycharmProjects\Deep-Modeling\data\nmos_40nm.csv')


from src.models.Regressor import Regressor
model = Regressor()

#from src.trainers.regressor import Regression_Trainer
#trainer = Regression_Trainer(n_epochs=100, weight_decay=1e-7)

#trainer.fit(a,model)
#trainer.score(a,model)

'''
from src.models.Classifier import Classifier
model_2 = Classifier()

from src.trainers.classifier_trainer import Classifier_Trainer
trainer = Classifier_Trainer(n_epochs=20)
trainer.fit(b,model_2)


'''

from src.model import Model
m = Model('regressor', a)
m.train()

m.save(r'model.pth')

m2 = Model('regressor',a)
m2.load('model.pth')
import numpy as np
tmp = np.array([0.4,0.5,100e-9])

m2(tmp)