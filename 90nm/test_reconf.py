import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import matplotlib.pyplot as plt
from src.datasets.dcop import *
from src.model import Model, Robust_model
from sklearn.ensemble import GradientBoostingRegressor
import os
from scipy import interpolate
from scipy.interpolate import interpn

path = "/home/kostastoulou/Documents/Deep_Modeling/models"
pmos_gm, pmos_gds = Robust_model(path=path,name='pmos_gm_TSMC_90GP',model_name='gm.joblib'), Robust_model(path=path,name='pmos_gds_TSMC_90GP',model_name='gds.joblib')
nmos_gm, nmos_gds = Robust_model(path=path,name='nmos_gm_TSMC_90GP',model_name='gm.joblib'), Robust_model(path=path,name='nmos_gds_TSMC_90GP',model_name='gds.joblib')

vdsn = []
vdsp= np.linspace(0.3,0.9,20)
for vdsp_ in vdsp:
    vdsn.append(1.3 -vdsp_)
vdsn = np.array(vdsn)
vgsn = vdsn
vgsp = vdsp
#pmos_gds(np.array([0.4,0.5,100*1e-9]).reshape(-1,3))

#vgsp = np.array([0.734]).reshape(-1,)
#vdsp = np.array([0.734]).reshape(-1,)
#vdsn, vgsn = 1.3- vdsp, 1.3-vgsp
L = np.array([100*1e-9]*20)


X = np.transpose(np.vstack((vgsn,vdsn,L)))
gmn = nmos_gm(X)
gdsn = nmos_gds(X)
X = np.transpose(np.vstack((vgsp,vdsp,L)))
gmp = pmos_gm(X)
gdsp = pmos_gds(X)


def blkm(L,VDS,VGS):
    vgs_temp = np.linspace(VGS * 0.8, VGS * 1.2, 16)
    vds_temp = np.ones(vgs_temp.shape)*VDS
    L = np.ones(vgs_temp.shape)*L[0]
    gm_, gm2_ = [],[]

    for i in range(vgs_temp.shape[1]):

        # load gm and gd values, they correspond to the above processing
        X = np.transpose(np.vstack((vgs_temp[:,i], vds_temp[:,i], L[:,i])))
        gm_data = nmos_gm(X)
        gd_data = nmos_gds(X)

        # so far temp is a 2-d vector with gm's for all vds,vgs combos
        GM2 =np.diff(gm_data, axis=0)/np.diff(vgs_temp[:,i])
        #GD2 = np.diff(gd_data)/np.transpose(np.diff(VDS))
        #X11 = np.transpose(np.transpose(np.diff(gd_data,axis=0))/np.diff(VGS))

        # the above derivatives are at the middlepoint of the (vds,vgs) tuple
        #vds1 = 0.5*(VDS[0:-1] + VDS[1:])
        vgs1 = 0.5*(vgs_temp[0:-1,i] + vgs_temp[1:,i])

        #gm2 = np.empty([len(VGS),len(VDS)])
        gm2 = interpolate.interp1d(vgs1,GM2,bounds_error=False,fill_value="extrapolate")(vgs_temp[:,i])
        #gd2 = np.empty([len(VGS),len(VDS)])
        #for ind,vgs in enumerate(VGS):
        #    gd2[ind,:] = interpolate.interp1d(vds1,GD2,bounds_error=False,fill_value="extrapolate")(VDS)
        #x11 = np.empty([len(VGS),len(VDS)])
        #for ind,vds in enumerate(VDS):
        #    x11[:,ind] = interpolate.interp1d(vgs1,X11,bounds_error=False,fill_value="extrapolate")(VGS)

        # interp2d to find the value at the given (vgs,vds) tuple
        #gm_2 = gm2[:,0]
        #gd_2 = gd2[0,:]
        #x_11= x11[:,0]

        ind = np.argmin(np.abs(vgs1 - VGS[i]))
        # return gm1,gd1,gm2,gd2,x11 for given (vds,vgs)
        gm_1 = nmos_gm(X)[ind]
        #gd_1 = nmos_gds(X)
        gm_2 = gm2[ind]
        gm_.append(gm_1)
        gm2_.append(gm_2)
        #gd_2 = gd_2(VGS,VDS).flatten()
        #x_11 = x_11(VGS,VDS).flatten()

    return np.array(gm_),np.array(gm2_)

gm_1n,gm_2n = blkm(L, vdsn, vgsn)
gm_1p,gm_2p = blkm(L, vdsp, vgsp)

print('Sum : {}'.format(gm_2n+gm_2p))
plt.plot(vgsn,gm_1p, label='pmos')
plt.plot(vgsn,gm_1n, label='nmos')
plt.plot(vgsn,gm_1n+gm_1p, label='Gm_1')
plt.legend()
plt.show()

plt.plot(vgsn,gm_2p, label='pmos')
plt.plot(vgsn,gm_2n, label='nmos')
plt.plot(vgsn,gm_2n+gm_2p, label='Gm_1')
plt.legend()
plt.show()


GM = gm_1n + gm_1p
ind = np.argmax(GM)
print(vgsn[ind])
'''
# blkm : L, VBS MUST EXIST !!!
vds = [0.77,0.9]
vgs = [0.3,0.4]
for v1,v2 in zip(vds,vgs):
    gm,gd,gm_2,gd_2,x_11 = blkm(240e-9, v1, v2, 0)
    print(gm,gd,gm_2,gd_2,x_11)
'''
