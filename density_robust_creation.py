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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

"""
Goal: create 4 models, (nmos, pmos), (jds,jds_lower)
"""

# nmos

path = r'/home/kostastoulou/Documents/automation/40nm_tables_v2/nmos_v4/nmos_40nm.csv'
df = pd.read_csv(path)
vgs, vds, region, L = df['vgs'].values, df['vds'].values, df['region'].values, df['L'].values
jds = df['jds'].values
jds_lower = df['jds_lower'].values

# preprocessing
X = np.transpose(np.vstack((vgs,vds,L)))
scaler = MinMaxScaler()
scaler.fit(X)
X_transformed = scaler.transform(X)

# fit
est = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.08,
    max_depth=10, random_state=0, loss='huber',min_samples_split=40).fit(X_transformed, jds)
y_pred = est.predict(X_transformed)
print(r2_score(jds,y_pred))
print(mean_squared_error(jds,y_pred))



'''
y_test_ind = (jds > 15) & (jds < 200)
y_test = jds[y_test_ind]
X_test_transformed = X_transformed[y_test_ind]
y_pred_test = est.predict(X_test_transformed)
diff = np.abs(y_test-y_pred_test)/y_test
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X_test_transformed[:,0], X_test_transformed[:,1], diff)
plt.show()
'''

est_lower = GradientBoostingRegressor(n_estimators=1200, learning_rate=0.08,
    max_depth=6, random_state=0, loss='huber',min_samples_split=40).fit(X_transformed, jds_lower)
y_pred = est_lower.predict(X_transformed)
print(r2_score(jds_lower,y_pred))
print(mean_squared_error(jds_lower, y_pred))

#saving
save_path = r'/home/kostastoulou/Documents/Deep_Modeling/models'
out_path_model = os.path.join(save_path,'robust_nmos_jds/jds.joblib')
out_path_model_2 = os.path.join(save_path,'robust_nmos_jds/jds_lower.joblib')
out_path_scaler =os.path.join(save_path,'robust_nmos_jds/scaler.joblib')
dump(est, out_path_model)
dump(est_lower, out_path_model_2)
dump(scaler, out_path_scaler)

# loading
loaded = load(out_path_model)
loaded_scaler = load(out_path_scaler)

X_scaled_loaded = loaded_scaler.transform(X)
y_pred = loaded.predict(X_scaled_loaded)
print(r2_score(jds,y_pred))



# pmos
path = '/home/kostastoulou/Documents/automation/40nm_tables_v2/pmos_v2/pmos_40nm_nosweep.csv'
df = pd.read_csv(path)
vgs, vds, region, L = df['vgs'].values, df['vds'].values, df['region'].values, df['L'].values
jds = df['jds'].values
jds_lower = df['jds_lower'].values

# preprocessing
X = np.transpose(np.vstack((vgs,vds,L)))
scaler = MinMaxScaler()
scaler.fit(X)
X_transformed = scaler.transform(X)

# fit
est = GradientBoostingRegressor(n_estimators=1200, learning_rate=0.08,
    max_depth=6, random_state=0, loss='huber',min_samples_split=40).fit(X_transformed, jds)
y_pred = est.predict(X_transformed)
print(r2_score(jds,y_pred))
est_lower = GradientBoostingRegressor(n_estimators=1200, learning_rate=0.08,
    max_depth=6, random_state=0, loss='huber',min_samples_split=40).fit(X_transformed, jds_lower)
y_pred = est_lower.predict(X_transformed)
print(r2_score(jds_lower,y_pred))


save_path = r'/home/kostastoulou/Documents/Deep_Modeling/models'
out_path_model = os.path.join(save_path,'robust_pmos_jds/jds.joblib')
out_path_model_2 = os.path.join(save_path,'robust_pmos_jds/jds_lower.joblib')
out_path_scaler =os.path.join(save_path,'robust_pmos_jds/scaler.joblib')
dump(est, out_path_model)
dump(est_lower, out_path_model_2)
dump(scaler, out_path_scaler)
