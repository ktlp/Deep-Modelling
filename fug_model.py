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
fug = df['fug'].values

print('min: {}, max:{}'.format(np.log10(min(fug)),np.log10(max(fug))))

# lets make fug --> log10(fug)
fug = np.log10(fug)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(fug.reshape((-1,1)))
fug_transformed = scaler.transform(fug.reshape((-1,1)))

X = np.transpose(np.vstack((vgs, vds, L)))
X_scaler = MinMaxScaler()
X_transformed = X_scaler.fit(X).transform(X)


from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=0, loss='huber').fit(X_transformed, fug_transformed.reshape((-1,)))


print(mean_squared_error(fug_transformed[:300], est.predict(X_transformed[:300])))
res = scaler.inverse_transform(est.predict(X_transformed).reshape((-1,1))).reshape((-1,))
true = scaler.inverse_transform(fug_transformed.reshape((-1,1))).reshape((-1,))
check = np.transpose(np.vstack((res.reshape((-1,)), true.reshape(-1,))))
check = np.power(10,check)
l = []
for i in range(check.shape[0]):
    l.append(abs((check[i,0] - check[i,1])/check[i,1]))

min(l)

import joblib
joblib.dump(est, '/home/kostastoulou/Documents/Deep_Modeling/fug_model.joblib')
joblib.dump(scaler, '/home/kostastoulou/Documents/Deep_Modeling/fug_scaler.joblib')
joblib.dump(X_scaler, '/home/kostastoulou/Documents/Deep_Modeling/fug_X_scaler.joblib')