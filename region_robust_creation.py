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



path = r'/home/kostastoulou/Documents/automation/40nm_tables_v2/nmos_v4/nmos_40nm.csv'
save_path = r'/home/kostastoulou/Documents/Deep_Modeling/models'

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

# preprocessing
df = pd.read_csv(path)
vgs, vds, region = df['vgs'].values, df['vds'].values, df['region'].values
L = df['L'].values
y = region.copy()
inds = np.where((y==0)|(y==1)|(y==3))
y[inds] = 0
y[y==2]=1
X = np.transpose(np.vstack((vgs,vds,L)))

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# training
clf = SVC(kernel='rbf' ,class_weight={1:8})
clf.fit(X_scaled,y)
y_pred = clf.predict(X_scaled)
print(confusion_matrix(y,y_pred))

# saving
import os
out_path_model = os.path.join(save_path,'robust_nmos_region/region.joblib')
out_path_scaler =os.path.join(save_path,'robust_nmos_region/scaler.joblib')
dump(clf, out_path_model)
dump(scaler, out_path_scaler)

# loading
loaded = load(out_path_model)
loaded_scaler = load(out_path_scaler)

# check
X_scaled_loaded = loaded_scaler.transform(X)
y_pred = loaded.predict(X_scaled_loaded)
print(confusion_matrix(y,y_pred))

## do the same for the pmos region !!!

path = '/home/kostastoulou/Documents/automation/40nm_tables_v2/pmos_v2/pmos_40nm_nosweep.csv'
df = pd.read_csv(path)
vgs, vds, region = df['vgs'].values, df['vds'].values, df['region'].values
L = df['L'].values
y = region.copy()
inds = np.where((y==0)|(y==1)|(y==3))
y[inds] = 0
y[y==2]=1
X = np.transpose(np.vstack((vgs,vds,L)))

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
clf = SVC(kernel='rbf' ,class_weight={1:10})
clf.fit(X_scaled,y)
y_pred = clf.predict(X_scaled)
print(confusion_matrix(y,y_pred))

out_path_model = os.path.join(save_path,'robust_pmos_region/region.joblib')
out_path_scaler =os.path.join(save_path,'robust_pmos_region/scaler.joblib')
dump(clf, out_path_model)
dump(scaler, out_path_scaler)

# loading
loaded = load(out_path_model)
loaded_scaler = load(out_path_scaler)

# check
X_scaled_loaded = loaded_scaler.transform(X)
y_pred = loaded.predict(X_scaled_loaded)
print(confusion_matrix(y,y_pred))

