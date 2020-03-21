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
df= df[df['L'] == df['L'].unique()[47]]
df_t =df[df['vgs'] < 0.52]
df_t =df_t[df_t['vgs'] > 0.46]
df_t = df_t[df_t['vds']<0.2]

from sklearn.metrics import confusion_matrix
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 1, 0, 1]
print(confusion_matrix(y_true, y_pred))