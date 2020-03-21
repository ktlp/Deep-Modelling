import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

path = r'/home/kostastoulou/Documents/automation/40nm_tables_v2/nmos_v4/nmos_40nm.csv'
df = pd.read_csv(path)
vgs, vds, region, L, cgs = df['vgs'].values, df['vds'].values, df['region'].values, df['L'].values, df['cgs'].values

print('Min of cgs: {}, max of cgs: {}'.format(min(cgs), max(cgs)))

cgg = df['cgg'].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(cgg.reshape((-1,1)))
cgg_transfomred = scaler.transform(cgg.reshape((-1,1)))

X = np.transpose(np.vstack((vgs, vds, L)))
X_scaler = MinMaxScaler()
X_transformed = X_scaler.fit(X).transform(X)

print('min: {}, max:{}'.format(min(cgg), max(cgg)))



from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators=700, learning_rate=0.1, max_depth=4, random_state=0, loss='huber').fit(X_transformed, cgg_transfomred.reshape((-1,)))


print(mean_squared_error(cgg_transfomred[:300], est.predict(X_transformed[:300])))
res = scaler.inverse_transform(est.predict(X_transformed).reshape((-1,1))).reshape((-1,))
true = scaler.inverse_transform(cgg_transfomred.reshape((-1,1))).reshape((-1,))
check = np.transpose(np.vstack((res.reshape((-1,)), true.reshape(-1,))))

l = []
for i in range(check.shape[0]):
    l.append(abs((check[i,0] - check[i,1])/check[i,1]))

min(l)

import joblib
joblib.dump(est, 'cgs_model.joblib')
joblib.dump(scaler, 'cgs_scaler.joblib')
joblib.dump(X_scaler, 'cgs_X_scaler.joblib')