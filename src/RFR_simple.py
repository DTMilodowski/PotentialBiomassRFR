"""
04/12/2018 - DTM
This file a single random forest fit to AGB data as a function of climate and
soil properties and land use.
"""

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

country_code = sys.argv[1]
version = sys.argv[2]

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'

pca = joblib.load('%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))

predictors,landmask = get_predictors(country_code, training_subset=True)

#transform the data
X = pca.transform(predictors)

#get the agb data
y = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0].values[landmask]

#split train and test subset, specifying random seed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=26)

#create the random forest object with predefined parameters
rf = RandomForestRegressor(n_jobs=20,random_state=26,
                            n_estimators = 1000,bootstrap=True,
                            min_samples_leaf = 20,max_features = 6)

rf.fit(X_train,y_train)

#save the fitted rf_grid
joblib.dump(rf,'%s/%s_%s_rf_single.pkl' % (path2alg,country_code,version))
