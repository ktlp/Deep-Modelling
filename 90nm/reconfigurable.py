import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import matplotlib.pyplot as plt
from src.datasets.dcop import *
from src.model import Model
from sklearn.ensemble import GradientBoostingRegressor
import os


'''

#########################
# GM MODELS
##########################

#this works well, for now
# create new model for gm for pmos
df = pd.read_csv(r'/home/kostastoulou/Documents/POC_optimizer/characterization/pmos90GP.csv')
df = df[df['region'] == 2]
X = np.transpose((np.vstack((df['vgs'].values, df['vds'].values, df['L'].values))))
y = df['gm'].values
scaler = MinMaxScaler()
X_transformed = scaler.fit(X).transform(X)
scaler2 = MinMaxScaler()
y_transformed = scaler2.fit(y.reshape(-1,1)).transform(y.reshape(-1,1))
gm2  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
    max_depth=8, random_state=0, loss='huber',min_samples_split=40).fit(X_transformed, y_transformed)
y_pred = gm2.predict(X_transformed)

save_path = r'/home/kostastoulou/Documents/Deep_Modeling/models'
out_path_model = os.path.join(save_path,'pmos_gm_TSMC_90GP/gm.joblib')
out_path_scaler =os.path.join(save_path,'pmos_gm_TSMC_90GP/scaler.joblib')
out_path_yscaler = os.path.join(save_path,'pmos_gm_TSMC_90GP/yscaler.joblib')
dump(gm2, out_path_model)
dump(scaler, out_path_scaler)
dump(scaler2, out_path_yscaler)

df = pd.read_csv(r'/home/kostastoulou/Documents/POC_optimizer/characterization/nmos90GP.csv')
df = df[df['region'] == 2]
X = np.transpose((np.vstack((df['vgs'].values, df['vds'].values, df['L'].values))))
y = df['gm'].values
scaler = MinMaxScaler()
X_transformed = scaler.fit(X).transform(X)
scaler2 = MinMaxScaler()
y_transformed = scaler2.fit(y.reshape(-1,1)).transform(y.reshape(-1,1))
gm2  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
    max_depth=8, random_state=0, loss='huber',min_samples_split=40).fit(X_transformed, y_transformed)
y_pred = gm2.predict(X_transformed)


save_path = r'/home/kostastoulou/Documents/Deep_Modeling/models'
out_path_model = os.path.join(save_path,'nmos_gm_TSMC_90GP/gm.joblib')
out_path_scaler =os.path.join(save_path,'nmos_gm_TSMC_90GP/scaler.joblib')
out_path_yscaler = os.path.join(save_path,'nmos_gm_TSMC_90GP/yscaler.joblib')
dump(gm2, out_path_model)
dump(scaler, out_path_scaler)
dump(scaler2, out_path_yscaler)

df = pd.read_csv('/home/kostastoulou/Documents/POC_optimizer/characterization/nmos90GP.csv')
df2 = df[df['region'] ==2]
vgs, vds, L, gds_vals, gm_vals = df2['vgs'].values, df2['vds'].values, df2['L'].values, df2['gds'].values, df2['gm'].values
X = np.transpose(np.vstack((vgs,vds,L)))

diff2 = np.abs(scaler2.inverse_transform(y_pred.reshape(-1,1)).reshape(-1,) - gm_vals)/gm_vals
plt.hist(diff2, bins=100)
plt.show()
inds = np.where(diff2>0.07)[0]
print(len(inds))
'''
################################
# GDS
################################

df = pd.read_csv(r'/home/kostastoulou/Documents/POC_optimizer/characterization/nmos90GP.csv')
df = df[df['region'] == 2]
X = np.transpose((np.vstack((df['vgs'].values, df['vds'].values, df['L'].values))))
y = df['gds'].values
scaler = MinMaxScaler()
X_transformed = scaler.fit(X).transform(X)
scaler2 = MinMaxScaler()
y_transformed = scaler2.fit(y.reshape(-1,1)).transform(y.reshape(-1,1))
gds  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
    max_depth=8, random_state=0, loss='huber',min_samples_split=40).fit(X_transformed, y_transformed)
y_pred = gds.predict(X_transformed)


save_path = r'/home/kostastoulou/Documents/Deep_Modeling/models'
out_path_model = os.path.join(save_path,'nmos_gds_TSMC_90GP/gds.joblib')
out_path_scaler =os.path.join(save_path,'nmos_gds_TSMC_90GP/scaler.joblib')
out_path_yscaler = os.path.join(save_path,'nmos_gds_TSMC_90GP/yscaler.joblib')
dump(gds, out_path_model)
dump(scaler, out_path_scaler)
dump(scaler2, out_path_yscaler)

df = pd.read_csv('/home/kostastoulou/Documents/POC_optimizer/characterization/nmos90GP.csv')
df2 = df[df['region'] ==2]
vgs, vds, L, gds_vals, gm_vals = df2['vgs'].values, df2['vds'].values, df2['L'].values, df2['gds'].values, df2['gm'].values
X = np.transpose(np.vstack((vgs,vds,L)))
diff2 = np.abs(scaler2.inverse_transform(y_pred.reshape(-1,1)).reshape(-1,) - gds_vals)/gds_vals
plt.hist(diff2, bins=100)
plt.show()
inds = np.where(diff2>0.07)[0]
print(len(inds))


df = pd.read_csv(r'/home/kostastoulou/Documents/POC_optimizer/characterization/pmos90GP.csv')
df = df[df['region'] == 2]
X = np.transpose((np.vstack((df['vgs'].values, df['vds'].values, df['L'].values))))
y = df['gds'].values
scaler = MinMaxScaler()
X_transformed = scaler.fit(X).transform(X)
scaler2 = MinMaxScaler()
y_transformed = scaler2.fit(y.reshape(-1,1)).transform(y.reshape(-1,1))
gds  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
    max_depth=8, random_state=0, loss='huber',min_samples_split=40).fit(X_transformed, y_transformed)
y_pred = gds.predict(X_transformed)


save_path = r'/home/kostastoulou/Documents/Deep_Modeling/models'
out_path_model = os.path.join(save_path,'pmos_gds_TSMC_90GP/gds.joblib')
out_path_scaler =os.path.join(save_path,'pmos_gds_TSMC_90GP/scaler.joblib')
out_path_yscaler = os.path.join(save_path,'pmos_gds_TSMC_90GP/yscaler.joblib')
dump(gds, out_path_model)
dump(scaler, out_path_scaler)
dump(scaler2, out_path_yscaler)

df = pd.read_csv('/home/kostastoulou/Documents/POC_optimizer/characterization/pmos90GP.csv')
df2 = df[df['region'] ==2]
vgs, vds, L, gds_vals, gm_vals = df2['vgs'].values, df2['vds'].values, df2['L'].values, df2['gds'].values, df2['gm'].values
X = np.transpose(np.vstack((vgs,vds,L)))
diff2 = np.abs(scaler2.inverse_transform(y_pred.reshape(-1,1)).reshape(-1,) - gds_vals)/gds_vals
plt.hist(diff2, bins=100)
plt.show()
inds = np.where(diff2>0.07)[0]
print(len(inds))

