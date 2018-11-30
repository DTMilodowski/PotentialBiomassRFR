"""
16/11/2018 - JFE
This files fits the best model from GridSearch to the whole dataset and saves
it for quick use in production files
 """

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import pandas as pd

country_code = sys.argv[1]

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms/'
path2calval = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/calval/'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'

#load the fitted  the fitted rf_grid and PCA
rf = joblib.load('%s%s_rf_grid.pkl' % (path2alg,country_code)).best_estimator_
pca = joblib.load('%s%s_pca_pipeline.pkl' % (path2alg,country_code))

#refit to whole dataset - get predictors and targets
predictors,landmask = get_predictors(country_code)
X = pca.transform(predictors)
med = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0].values[landmask]
unc = xr.open_rasterio('%sAvitabile_AGB_Uncertainty_%s_1km.tif' % (path2agb,country_code))[0].values[landmask]

lvls='mean'
agb[agb<0] = 0
rf.fit(X,agb)
print(mean_squared_error(rf.predict(X),agb))
joblib.dump(rf,'%s%s_rf_mean.pkl' % (path2alg,country_code))
"""
# commented out for now - code for +/-uncertainty levels
lvls = ['mean','upper','lower']
for aa, agb in enumerate([med,med+unc,med-unc]):
    agb[agb<0] = 0
    rf.fit(X,agb)
    print(mean_squared_error(rf.predict(X),agb))
    joblib.dump(rf,'/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/rf_%s.pkl' % lvls[aa])
"""
