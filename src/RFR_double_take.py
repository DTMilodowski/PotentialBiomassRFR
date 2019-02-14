"""
04/12/2018 - DTM
This file a single random forest fit to AGB data as a function of climate and
soil properties and land use.
"""

from useful import *
import set_training_areas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

country_code = 'INDO'#sys.argv[1]
version = '001'#sys.argv[2]

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'

pca = joblib.load('%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))

predictors,trainmask = get_predictors(country_code, training_subset=True)

#transform the data
X = pca.transform(predictors)

#get the agb data
agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0]
y = agb.values[trainmask]

#create the random forest object with predefined parameters
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=20, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
           oob_score=True, random_state=None, verbose=0, warm_start=False)

#rf.fit(X,y)

#save the fitted rf_grid
#joblib.dump(rf,'%s/%s_%s_rf_single_pass1.pkl' % (path2alg,country_code,version))
rf = joblib.load('%s/%s_%s_rf_single_pass1.pkl' % (path2alg,country_code,version))
#------------------------------------
# 2nd pass
# fit the full dataset
predictors,landmask = get_predictors(country_code, training_subset=False)
Xall = pca.transform(predictors)
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

trainmask2, trainflag = set_training_areas.set_revised(path2data,agb_rf.AGBobs.values,agb_rf.AGBpot1.values,landmask)
y2 = agb.values[trainmask2]
X2 = Xall[trainmask2[landmask]]

#split train and test subset, specifying random seed
X_train, X_test, y_train, y_test = train_test_split(X2,y2,test_size = 0.25, random_state=26)
rf.fit(X_train,y_train)

joblib.dump(rf,'%s/%s_%s_rf_single_pass2.pkl' % (path2alg,country_code,version))
rf = joblib.load('%s/%s_%s_rf_single_pass2.pkl' % (path2alg,country_code,version))


agb_rf.AGBpot2.values[landmask] = rf.predict(Xall)

# pass 3
trainmask3, trainflag = set_training_areas.set_revised(path2data,agb_rf.AGBobs.values,agb_rf.AGBpot2.values,landmask)
y3= agb.values[trainmask3]
X3= Xall[trainmask3[landmask]]

#split train and test subset, specifying random seed
X_train, X_test, y_train, y_test = train_test_split(X3,y3,test_size = 0.25, random_state=26)
rf.fit(X_train,y_train)

joblib.dump(rf,'%s/%s_%s_rf_single_pass3.pkl' % (path2alg,country_code,version))
rf = joblib.load('%s/%s_%s_rf_single_pass3.pkl' % (path2alg,country_code,version))

agb_rf.AGBpot3.values[landmask] = rf.predict(Xall)


# pass 4
trainmask4, trainflag = set_training_areas.set_revised(path2data,agb_rf.AGBobs.values,agb_rf.AGBpot3.values,landmask)
y4= agb.values[trainmask4]
X4= Xall[trainmask4[landmask]]
rf.fit(X4,y4)
joblib.dump(rf,'%s/%s_%s_rf_single_pass4.pkl' % (path2alg,country_code,version))
rf = joblib.load('%s/%s_%s_rf_single_pass4.pkl' % (path2alg,country_code,version))
AGBpot4=np.zeros([ny,nx])
AGBpot4[landmask] = rf.predict(Xall)

# pass 4
trainmask5, trainflag = set_training_areas.set_revised(path2data,agb_rf.AGBobs.values,AGBpot4,landmask)
y5= agb.values[trainmask5]
X5= Xall[trainmask5[landmask]]
rf.fit(X5,y5)
joblib.dump(rf,'%s/%s_%s_rf_single_pass5.pkl' % (path2alg,country_code,version))
rf=joblib.load('%s/%s_%s_rf_single_pass5.pkl' % (path2alg,country_code,version))
AGBpot5=np.zeros([ny,nx])
AGBpot5[landmask] = rf.predict(Xall)



# pass 6
trainmask6, trainflag = set_training_areas.set_revised(path2data,agb_rf.AGBobs.values,AGBpot5,landmask)
y6= agb.values[trainmask6]
X6= Xall[trainmask6[landmask]]
rf.fit(X6,y6)
joblib.dump(rf,'%s/%s_%s_rf_single_pass6.pkl' % (path2alg,country_code,version))
AGBpot6=np.zeros([ny,nx])
AGBpot6[landmask] = rf.predict(Xall)
"""
# pass 7
trainmask7, trainflag = set_training_areas.set_revised(path2data,agb_rf.AGBobs.values,AGBpot6,landmask)
y7= agb.values[trainmask7]
X7= Xall[trainmask7[landmask]]
rf.fit(X7,y7)
joblib.dump(rf,'%s/%s_%s_rf_single_pass7.pkl' % (path2alg,country_code,version))
AGBpot7=np.zeros([ny,nx])
AGBpot7[landmask] = rf.predict(Xall)
# pass 7
trainmask8, trainflag = set_training_areas.set_revised(path2data,agb_rf.AGBobs.values,AGBpot7,landmask)
y8= agb.values[trainmask8]
X8= Xall[trainmask8[landmask]]
rf.fit(X8,y8)
joblib.dump(rf,'%s/%s_%s_rf_single_pass8.pkl' % (path2alg,country_code,version))
AGBpot8=np.zeros([ny,nx])
AGBpot8[landmask] = rf.predict(Xall)
# pass 7
trainmask9, trainflag = set_training_areas.set_revised(path2data,agb_rf.AGBobs.values,AGBpot8,landmask)
y9= agb.values[trainmask9]
X9= Xall[trainmask9[landmask]]
rf.fit(X9,y9)
joblib.dump(rf,'%s/%s_%s_rf_single_pass9.pkl' % (path2alg,country_code,version))
AGBpot9=np.zeros([ny,nx])
AGBpot9[landmask] = rf.predict(Xall)
"""

n_training_pixels = np.zeros(6)
n_training_pixels[0] = np.sum(trainmask)
n_training_pixels[1] = np.sum(trainmask2)
n_training_pixels[2] = np.sum(trainmask3)
n_training_pixels[3] = np.sum(trainmask4)
n_training_pixels[4] = np.sum(trainmask5)
n_training_pixels[5] = np.sum(trainmask6)
"""
n_training_pixels[6] = np.sum(trainmask7)
n_training_pixels[7] = np.sum(trainmask8)
n_training_pixels[8] = np.sum(trainmask9)
"""

agbpot = np.zeros(5)
agbpot[0] = np.nansum(agb_rf.AGBpot1.values[landmask])
agbpot[1] = np.nansum(agb_rf.AGBpot2.values[landmask])
agbpot[2] = np.nansum(agb_rf.AGBpot3.values[landmask])
agbpot[3] = np.nansum(AGBpot4[landmask])
agbpot[4] = np.nansum(AGBpot5[landmask])
agbpot[5] = np.nansum(AGBpot6[landmask])
"""
agbpot[6] = np.nansum(AGBpot7[landmask])
agbpot[7] = np.nansum(AGBpot8[landmask])
agbpot[8] = np.nansum(AGBpot9[landmask])
"""

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
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
#save to a nc file
encoding = {'AGB_mean':{'zlib':True,'complevel':1},
            'AGB_upper':{'zlib':True,'complevel':1},
            'AGB_lower':{'zlib':True,'complevel':1},}
agb_rf.to_netcdf('%s%s_%s_AGB_potential_RFR_worldclim_soilgrids_2pass.nc' % (path2output,country_code,version),encoding=encoding)
"""
