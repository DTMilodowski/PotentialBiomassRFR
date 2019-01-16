"""
04/12/2018 - DTM
This file a single random forest fit to AGB data as a function of climate and
soil properties and land use.
"""

from useful import *
import set_training_areas
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import forestci as fci # Wager et al. 2014 confidence intervals for scikit-learn RF

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

country_code = 'WAFR'#sys.argv[1]
version = '002'#sys.argv[2]
iterations = 5#int[sys.argv[3]]
load = 'new'#sys.argv[4]

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'

pca = joblib.load('%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))

# load all predictors to generate preprocessing minmax scalar transformation
predictors_full,landmask = get_predictors(country_code, training_subset=False)
# Get full land mask
Xall = pca.transform(predictors_full)
scaler = MinMaxScaler()
scaler.fit(Xall)
Xall = scaler.transform(Xall)

# load predictors for trainings data only
predictors,trainmask = get_predictors(country_code, training_subset=True)

#transform the data
X = scaler.transform(pca.transform(predictors))

#get the agb data
agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0]

yscaler = MinMaxScaler()
yscaler.fit(agb.values[trainmask].reshape(-1, 1))
y = yscaler.transform(agb.values[trainmask].reshape(-1, 1))
back_correction_of_variance = (np.max(agb.values[trainmask])-np.min(agb.values[trainmask]))**2.


#create the random forest object with predefined parameters
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=20, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
           oob_score=True, random_state=None, verbose=0, warm_start=False)

if load=='load':
    rf = joblib.load('%s/%s_%s_rf_single_pass1.pkl' % (path2alg,country_code,version))
else:
    rf.fit(X,y)
    joblib.dump(rf,'%s/%s_%s_rf_single_pass1.pkl' % (path2alg,country_code,version))

# set up some arrays
n_training_pixels = np.zeros(iterations)
agbpot = np.zeros(iterations)*np.nan
AGBpot = np.zeros((iterations,agb.values.shape[0],agb.values.shape[1]))*np.nan
error = np.zeros((iterations,agb.values.shape[0],agb.values.shape[1]))*np.nan
variance_IJ_unbiased = np.zeros((iterations,agb.values.shape[0],agb.values.shape[1]))*np.nan

"""
# get variance and error
variance_IJ_unbiased[0][landmask] = fci.random_forest_error(rf, X, Xall)
error[0] = np.sqrt(variance_IJ_unbiased[0])

# 1st iteration
AGBpot[0][landmask] = rf.predict(Xall)
n_training_pixels[0] = np.sum(trainmask)
agbpot[0] = np.nansum(AGBpot[0][landmask])
# get variance and error
variance_IJ_unbiased[0][landmask] = fci.random_forest_error(rf, X, Xall)
error[0] = np.sqrt(variance_IJ_unbiased[0])

"""

# getting variance is memory and processor intensive - restrict next phase to the
# additional forest only
# - get additional stable forest areas
other_stable_forest = set_training_areas.get_stable_forest_outside_training_areas(path2data,trainmask,landmask)
predictors_osf = get_predictors_for_defined_mask(country_code,other_stable_forest)
Xosf = scaler.transform(pca.transform(predictors_osf))


# 1st iteration
AGBpot[0][landmask] = yscaler.inverse_transform(rf.predict(Xall))
n_training_pixels[0] = np.sum(trainmask)
agbpot[0] = np.nansum(AGBpot[0][landmask])
# get variance and error
variance_IJ_unbiased[0][other_stable_forest] = back_correction_of_variance*fci.random_forest_error(rf, X, Xosf, memory_constrained = True, memory_limit = 20000)
error[0] = np.sqrt(variance_IJ_unbiased[0])

# subsequent iterations
for ii in range(1,iterations):
    trainmask, trainflag = set_training_areas.set_revised(path2data,agb.values,AGBpot[ii-1]+error[ii-1],landmask)
    yii = yscaler.transform(agb.values[trainmask].reshape(-1, 1))
    Xii = Xall[trainmask[landmask]]
    if load=='load':
        rf = joblib.load('%s/%s_%s_rf_single_pass%i.pkl' % (path2alg,country_code,version,ii+1))
    else:
        rf.fit(Xii,yii)
        joblib.dump(rf,'%s/%s_%s_rf_single_pass%i.pkl' % (path2alg,country_code,version,ii+1))
    # fit model
    AGBpot[ii][landmask] = yscalar.inverse_transform(rf.predict(Xall))
    variance_IJ_unbiased[ii][landmask] = back_correction_of_variance*fci.random_forest_error(rf, Xii, Xall)
    error[ii] = np.sqrt(variance_IJ_unbiased[ii])

    # summaries
    n_training_pixels[ii] = np.sum(trainmask)
    agbpot[ii] = np.nansum(AGBpot[ii][landmask])


"""
# xaray stuff for future reference
tr = agb.attrs['transform']#Affine.from_gdal(*agb.attrs['transform'])
transform = Affine(tr[0],tr[1],tr[2],tr[3],tr[4],tr[5])
nx, ny = agb.sizes['x'], agb.sizes['y']
col,row = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5)
lon, lat = transform * (col,row)

coords = {'lat': (['lat'],lat[:,0],{'units':'degrees_north','long_name':'latitude'}),
          'lon': (['lon'],lon[0,:],{'units':'degrees_east','long_name':'longitude'})}

attrs={'_FillValue':-9999.,'units':'Mg ha-1'}
data_vars = {}

data_vars['AGBobs'] = (['lat','lon'],np.zeros([ny,nx])-9999.,attrs)
data_vars['AGBpot1'] = (['lat','lon'],np.zeros([ny,nx])-9999.,attrs)
data_vars['AGBpot2'] = (['lat','lon'],np.zeros([ny,nx])-9999.,attrs)
data_vars['AGBpot3'] = (['lat','lon'],np.zeros([ny,nx])-9999.,attrs)

agb_rf = xr.Dataset(data_vars=data_vars,coords=coords)
agb_rf.AGBobs.values[landmask]  = agb.values[landmask]
agb_rf.AGBpot1.values[landmask] = rf.predict(Xall)

#save to a nc file
encoding = {'AGB_mean':{'zlib':True,'complevel':1},
            'AGB_upper':{'zlib':True,'complevel':1},
            'AGB_lower':{'zlib':True,'complevel':1},}
agb_rf.to_netcdf('%s%s_%s_AGB_potential_RFR_worldclim_soilgrids_2pass.nc' % (path2output,country_code,version),encoding=encoding)


"""
"""
# some plotting stuff

plt.cla()
plt.subplot(4,2,3); plt.imshow(trainmask)#;plt.colorbar()
plt.subplot(4,2,5); plt.imshow(trainmask2)#;plt.colorbar()
plt.subplot(4,2,7); plt.imshow(trainmask3)#;plt.colorbar()
plt.subplot(4,2,2); plt.imshow(agb_rf.AGBobs.values,vmin=0,vmax=350);plt.colorbar()
plt.subplot(4,2,4); plt.imshow(agb_rf.AGBpot1.values,vmin=0,vmax=350);plt.colorbar()
plt.subplot(4,2,6); plt.imshow(agb_rf.AGBpot2.values,vmin=0,vmax=350);plt.colorbar()
plt.subplot(4,2,8); plt.imshow(agb_rf.AGBpot3.values,vmin=0,vmax=350);plt.colorbar()
plt.show()


plt.subplot(3,2,1); plt.imshow(agb_rf.AGBobs.values,vmin=0,vmax=350);plt.colorbar()
plt.subplot(3,2,3); plt.imshow(agb_rf.AGBpot1.values,vmin=0,vmax=350);plt.colorbar()
plt.subplot(3,2,5); plt.imshow(agb_rf.AGBpot2.values,vmin=0,vmax=350);plt.colorbar()
plt.subplot(3,2,2); plt.imshow(agb_rf.AGBpot3.values,vmin=0,vmax=350);plt.colorbar()
plt.subplot(3,2,4); plt.imshow(AGBpot4,vmin=0,vmax=350);plt.colorbar()
plt.subplot(3,2,6); plt.imshow(AGBpot5,vmin=0,vmax=350);plt.colorbar()
plt.show()

ref = agb_rf.AGBobs.values
plt.subplot(3,2,1); plt.imshow(agb_rf.AGBobs.values-ref,vmin=-100,vmax=100,cmap='bwr');plt.colorbar()
plt.subplot(3,2,3); plt.imshow(agb_rf.AGBpot1.values-ref,vmin=-100,vmax=100,cmap='bwr');plt.colorbar()
plt.subplot(3,2,5); plt.imshow(agb_rf.AGBpot2.values-ref,vmin=-100,vmax=100,cmap='bwr');plt.colorbar()
plt.subplot(3,2,2); plt.imshow(agb_rf.AGBpot3.values-ref,vmin=-100,vmax=100,cmap='bwr');plt.colorbar()
plt.subplot(3,2,4); plt.imshow(AGBpot6-ref,vmin=-100,vmax=100,cmap='bwr');plt.colorbar()
plt.subplot(3,2,6); plt.imshow(AGBpot7-ref,vmin=-100,vmax=100,cmap='bwr');plt.colorbar()
plt.show()

agb1 = np.zeros(agb.values.shape)*np.nan
agb2 = agb1.copy()
agb3 = agb1.copy()
agb4 = agb1.copy()
agb5 = agb1.copy()
agb6 = agb1.copy()
agb7 = agb1.copy()
agb9 = agb1.copy()
agb2[trainmask2]=agb.values[trainmask2]
agb3[trainmask3]=agb.values[trainmask3]
agb4[trainmask4]=agb.values[trainmask4]
agb5[trainmask5]=agb.values[trainmask5]


plt.cla()
plt.subplot(2,2,1); plt.imshow(agb2,vmin=0,vmax=350,cmap='plasma')#;plt.colorbar()
plt.subplot(2,2,3); plt.imshow(agb3,vmin=0,vmax=350,cmap='plasma')#;plt.colorbar()
plt.subplot(2,2,2); plt.imshow(agb4,vmin=0,vmax=350,cmap='plasma')#;plt.colorbar()
plt.subplot(2,2,4); plt.imshow(agb5,vmin=0,vmax=350,cmap='plasma')#;plt.colorbar()
plt.show()

plt.cla()
plt.subplot(2,2,1); plt.imshow(agb_rf.AGBpot2.values,vmin=0,vmax=350,cmap='plasma')#;plt.colorbar()
plt.subplot(2,2,3); plt.imshow(agb_rf.AGBpot3.values,vmin=0,vmax=350,cmap='plasma')#;plt.colorbar()
plt.subplot(2,2,2); plt.imshow(AGBpot4,vmin=0,vmax=350,cmap='plasma')#;plt.colorbar()
plt.subplot(2,2,4); plt.imshow(AGBpot5,vmin=0,vmax=350,cmap='plasma')#;plt.colorbar()
plt.show()

import pandas as pd
df = pd.DataFrame({'obs':agb.values[trainmask5],'sim':AGBpot5[trainmask5]})
df.sim[df.sim<0] = 0.
fig = plt.figure('cal/val random',figsize=(10,6))
ax = fig.add_subplot(1,1,1,aspect='equal')
sns.regplot(x='obs',y='sim',data=df.sample(5000),scatter_kws={'s':1},line_kws={'color':'k'},ax=ax)

#adjust style
ax.set_title('(n = %05i)' % df.shape[0])
plt.xlim(0,550);plt.ylim(0,550)
plt.xlabel('AGB from Avitabile et al. (2016) [Mg ha $^{-1}$]')
plt.ylabel('Reconstructed AGB [Mg ha $^{-1}$]')

#show / save
fig.show()


"""
