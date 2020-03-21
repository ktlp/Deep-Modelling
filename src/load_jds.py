from src.datasets.dcop import *
from src.model import Model

path = r'/home/kostastoulou/Documents/automation/40nm_tables_v2/nmos_v4/nmos_40nm.csv'
path_pmos =  r'/home/kostastoulou/Documents/automation/40nm_tables_v2/nmos_v4/nmos_40nm.csv'
models_path = '/home/kostastoulou/Documents/Deep_Modeling/models/'

# vgs vds L
nmos_jds_dataset, nmos_jds_lower_dataset = Jds_dataset(path), Jds_dataset(path, label = 'jds_lower')
pmos_jds_dataset, pmos_jds_lower_dataset = Jds_dataset(path), Jds_dataset(path, label = 'jds_lower')
nmos_jds, nmos_jds_lower = Model('regressor',nmos_jds_dataset), Model('regressor',nmos_jds_lower_dataset)
pmos_jds, pmos_jds_lower = Model('regressor',pmos_jds_dataset), Model('regressor',pmos_jds_lower_dataset)

nmos_jds.load(models_path,'jds')
nmos_jds_lower.load(models_path, 'jds_lower')
pmos_jds_lower.load(models_path, 'pmos_jds_lower')
pmos_jds.load(models_path, 'pmos_jds')
