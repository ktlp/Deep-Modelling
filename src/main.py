import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = r'/home/kostastoulou/Documents/Deep_Modeling/logs/log.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

from src.datasets.jds import Jds_dataset
from src.datasets.region import Region_dataset
a = Jds_dataset(r'/home/kostastoulou/Documents/automation/40nm_tables_v2/nmos_v4/nmos_40nm.csv')
b =  Region_dataset(r'/home/kostastoulou/Documents/automation/40nm_tables_v2/nmos_v4/nmos_40nm.csv')


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
#m = Model('classifier', b)
#m.train()
#m = Model('regressor', a)
#m.train()

#m.save(r'model.pth')

m2 = Model('classifier',b)
m2.load('model.pth')
import numpy as np
tmp = np.array([0.5,0.5,120e-9])

print(m2(tmp))

