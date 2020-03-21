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

from src.datasets.dcop import *

path = r'/home/kostastoulou/Documents/automation/40nm_tables_v2/nmos_v4/nmos_40nm.csv'


a = Jds_dataset(path)
a_lower = Jds_dataset(path, label = 'jds_lower')
#b =  Region_dataset(path)
#c = Vdsat_dataset(path)
#d = Vth_dataset(path)

pmos_jds = Jds_dataset('/home/kostastoulou/Documents/automation/40nm_tables_v2/pmos_v2/pmos_40nm_nosweep.csv')
pmos_jds_lower = Jds_dataset('/home/kostastoulou/Documents/automation/40nm_tables_v2/pmos_v2/pmos_40nm_nosweep.csv', label = 'jds_lower')

pmos_region = Region_dataset('/home/kostastoulou/Documents/automation/40nm_tables_v2/pmos_v2/pmos_40nm_nosweep.csv')

from src.model import Model
save_path = r'/home/kostastoulou/Documents/Deep_Modeling/models'


jds = Model('regressor', a)
jds_lower = Model('regressor', a_lower)
jds.train()
jds.save(save_path,'nmos_jds_full')
jds_lower.train()
jds_lower.save(save_path,'nmos_jds_lower_full')

''':arg

jds = Model('regressor', a)
jds.train()
jds.save(save_path, 'jds')
jds_lower = Model('regressor', a_lower)
jds_lower.train()
jds_lower.save(save_path,'jds_lower')
region = Model('classifier', b)
region.train()
region.save(save_path, 'region')
vdsat = Model('regressor', c)
vdsat.train()
vdsat.save(save_path, 'vdsat')
vth = Model('regressor', d)
vth.train()
vth.save(save_path, 'vth')

'''

#m = Model('classifier', b)
#m.train()
#m = Model('regressor', a)
#m.train()
#m.save(r'model.pth')
'''
from sklearn.linear_model import LogisticRegression
X, y = b.train_set.x_data.numpy()[:,:2],b.train_set.y_data.numpy()
X_test, y_test = b.test_set.x_data.numpy()[:,:2],b.test_set.y_data.numpy()
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.score(X_test,y_test))

import numpy as np
import matplotlib.pyplot as plt

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .05, X[:, 0].max() + .05
y_min, y_max = X[:, 1].min() - .05, X[:, 1].max() + .05
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.show()
'''
