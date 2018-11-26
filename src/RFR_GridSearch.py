"""
14/11/2018 - JFE
This files performs a grid search on random forest hyperparameters to fit
AGB data as a function of climate, soil properties and land use.
Parameters have been chosen after an initial RandomizedSearch highlighted
the most sensitive parameters.
"""

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

pca = joblib.load('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/pca_pipeline.pkl')

predictors,landmask = get_predictors(y0=2000,y1=2009)

#transform the data
X = pca.transform(predictors)

#get the agb data
y = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')[0].values[landmask]

#split train and test subset, specifying random seed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=26)

#define the parameters for the gridsearch
param_grid = { "max_features": np.linspace(.35,.7,8), "min_samples_leaf": np.linspace(1,10,10,dtype='i')}

#create the random forest object with predefined parameters
rf = RandomForestRegressor(n_jobs=20,random_state=26,
                            n_estimators = 1000,bootstrap=True)

#perform a grid search on hyper parameters using training subset of data
rf_grid = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,
                            verbose = 3,scoring = 'neg_mean_squared_error', n_jobs=1)

rf_grid.fit(X_train,y_train)

#save the fitted rf_grid
joblib.dump(rf_grid,'/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/rf_grid.pkl',compress=1)
