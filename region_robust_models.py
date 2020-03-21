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
nmos_region =  Region_dataset(path)
pmos_region = Region_dataset('/home/kostastoulou/Documents/automation/40nm_tables_v2/pmos_v2/pmos_40nm_nosweep.csv')

from src.model import Model
save_path = r'/home/kostastoulou/Documents/Deep_Modeling/models'

import pandas as pd
df = pd.read_csv(path)

# keep df entries for L eq 180n
print(df['L'].unique()[47])
df_2 = df[df['L'] == df['L'].unique()[47]]
ax1 = df_2.plot.scatter(x='vgs',y='vds',c='region',colormap='viridis')
vgs, vds, region = df_2['vgs'].values, df_2['vds'].values, df_2['region'].values
import matplotlib.pyplot as plt
plt.scatter(vgs,vds, c=region, s=0.1, marker='.')
plt.title('180')
plt.show()

# keep df entries for L eq 130n
print(df['L'].unique()[30])
df_3 = df[df['L'] == df['L'].unique()[30]]
ax2 = df_3.plot.scatter(x='vgs',y='vds',c='region',colormap='viridis')
vgs, vds, region = df_3['vgs'].values, df_3['vds'].values, df_3['region'].values
import matplotlib.pyplot as plt
plt.scatter(vgs,vds, c=region,  marker='.', s=0.1)
plt.title('130')
plt.show()

# lets fit a basic sklearn model to 130n dataset
import numpy as np
from sklearn.svm import SVC
clf = SVC(gamma='auto')
X= np.transpose(np.vstack((vgs,vds)))
y = region
print(np.unique(y))
inds = np.where((y == 0) | (y == 1) | (y == 3))
y[inds] = 0
y[y==2] =1
print(np.unique(y))
clf.fit(X,y)
y_pred = clf.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)
'''
out:
array([[2520,   96],
       [ 130, 2504]])
'''
# I want to eliminate those 96 misclassifications
# lets fit other svms
clf_linear = SVC(kernel='rbf' ,class_weight={0:1.5})
clf_linear.fit(X,y)
y_pred = clf_linear.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)
plt.scatter(X[:,0],X[:,1], c=y,  marker='.', s=0.2)
plt.title('130')
plt.show()
'''
output:
array([[2616,    0],
       [  66, 2568]])
'''

# let's try to fit the same model to the whole dataset
vgs, vds, region = df['vgs'].values, df['vds'].values, df['region'].values
L = df['L'].values
y = region.copy()
inds = np.where((y==0)|(y==1)|(y==3))
y[inds] = 0
y[y==2]=1
np.unique(y)
X = np.transpose(np.vstack((vgs,vds,L)))
clf_linear = SVC(kernel='rbf' ,class_weight={0:1.5})
clf_linear.fit(X,y)
y_pred = clf_linear.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)
'''
array([[155202,   2104],
       [  3858, 153836]])
'''
# too much type 1 errors!
# lets try scaling first
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaler = scaler.fit(X)
X_scaled = scaler.transform(X)
clf_linear = SVC(kernel='rbf' ,class_weight={0:1.5})
clf_linear.fit(X_scaled,y)
y_pred = clf_linear.predict(X_scaled)
print(confusion_matrix(y,y_pred))
'''
output:
[[157124    182]
 [  1679 156015]]
'''
# much better results!
# now let increase the weight
clf_linear = SVC(kernel='rbf' ,class_weight={0:3})
clf_linear.fit(X_scaled,y)
y_pred = clf_linear.predict(X_scaled)
print(confusion_matrix(y,y_pred))
'''
output:
[[157269     37]
 [  3297 154397]]
'''
# good results, 99% acciracy
# let's increase the class weight a bit more:
class_weights = [8,9]
for weight in class_weights:
	clf_linear = SVC(kernel='rbf', class_weight={0: weight})
	clf_linear.fit(X_scaled, y)
	y_pred = clf_linear.predict(X_scaled)
	conf = confusion_matrix(y, y_pred)
	type_1 = conf[0,1]
	acc = (conf[0,1]+conf[1,0])/(conf[0,1]+conf[1,0]+conf[1,1]+conf[0,0])
	print('Class weight : {} Missclassification Rate : {} Type 1 errors : {}'.format(weight,acc,type_1))

