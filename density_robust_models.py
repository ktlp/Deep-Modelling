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
from mpl_toolkits.mplot3d import Axes3D

path = r'/home/kostastoulou/Documents/automation/40nm_tables_v2/nmos_v4/nmos_40nm.csv'
df = pd.read_csv(path)

# let's keep only one L
df_2 = df[df['L'] == df['L'].unique()[40]]
vgs, vds, region = df_2['vgs'].values, df_2['vds'].values, df_2['region'].values
jds = df_2['jds'].values
jds_lower = df_2['jds_lower'].values

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, jds)
plt.show()

X = np.transpose(np.vstack((vgs,vds)))
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=10)
poly.fit(X)
X_transformed = poly.transform(X)

from sklearn.linear_model import Lasso
reg = Lasso(alpha=.2, max_iter=2000)
reg.fit(X_transformed,jds)

y_pred = reg.predict(X_transformed)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, y_pred)
plt.show()

# let's scale the output
from sklearn.preprocessing import MinMaxScaler
scaler_2 = MinMaxScaler()
scaler_2.fit(jds.reshape(-1,1))
y_transformed = scaler_2.transform(jds.reshape(-1,1))
reg.fit(X_transformed,y_transformed)

y_pred = reg.predict(X_transformed)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, y_pred)
plt.show()

# no good
# let's compute the error
reg = Lasso(alpha=.2, max_iter=2000)
reg.fit(X_transformed,jds)

y_pred = reg.predict(X_transformed)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, y_pred)
plt.show()

from sklearn.metrics import r2_score
r2_score_lasso = r2_score(jds, y_pred)

# overlay
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, y_pred, c='r')
ax.scatter(vgs, vds, jds, c='b')
plt.show()

# difference
fig = plt.figure()
diff = np.abs(y_pred-jds)/(jds+3)
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, diff, c='r')
plt.show()

# bagging
from sklearn.ensemble import BaggingRegressor
regr = BaggingRegressor(base_estimator=Lasso(alpha=.1, max_iter=2000),
                        n_estimators=20, random_state=0).fit(X_transformed, jds)
y_pred = regr.predict(X_transformed)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, y_pred)
plt.show()
# difference
fig = plt.figure()
diff = np.abs(y_pred-jds)/(jds+1)
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, diff, c='r')
plt.show()

# let's reduce the model's deggrees
poly = PolynomialFeatures(degree=3)
poly.fit(X)
X_transformed = poly.transform(X)
from sklearn.ensemble import BaggingRegressor
regr = BaggingRegressor(base_estimator=Lasso(alpha=.05, max_iter=2000),
                        n_estimators=100, random_state=0).fit(X_transformed, jds)
y_pred = regr.predict(X_transformed)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, y_pred)
plt.show()
# difference
fig = plt.figure()
diff = np.abs(y_pred-jds)/(jds+1)
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, diff, c='r')
plt.show()

# gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
    max_depth=3, random_state=0, loss='huber').fit(X_transformed, jds)
y_pred = est.predict(X_transformed)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, y_pred, c='r')
ax.scatter(vgs, vds, y_pred, c='b')

plt.show()
# difference
fig = plt.figure()
diff = np.abs(y_pred-jds)/(jds+1)
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, diff, c='r')
plt.show()
from sklearn.metrics import r2_score
print(r2_score(jds,y_pred))


# now with no transformation..
est = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
    max_depth=3, random_state=0, loss='huber').fit(X, jds)
y_pred = est.predict(X)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, y_pred, c='r')
ax.scatter(vgs, vds, y_pred, c='b')

plt.show()
# difference
fig = plt.figure()
diff = np.abs(y_pred-jds)/(jds+1)
ax = fig.gca(projection='3d')
ax.scatter(vgs, vds, diff, c='r')
plt.show()
from sklearn.metrics import r2_score
print(r2_score(jds,y_pred))

# let's try the gradient boosting algorithm for the full dataset
vgs, vds, region, L = df['vgs'].values, df['vds'].values, df['region'].values, df['L'].values
jds = df['jds'].values
jds_lower = df['jds_lower'].values

# preprocessing
X = np.transpose(np.vstack((vgs,vds,L)))
scaler = MinMaxScaler()
scaler.fit(X)
X_transformed = scaler.transform(X)

# fit
est = GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
    max_depth=5, random_state=0, loss='lad',min_samples_split=30).fit(X_transformed, jds)
y_pred = est.predict(X_transformed)
print(r2_score(jds,y_pred))



