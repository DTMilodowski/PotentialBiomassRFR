"""
13/11/2018 - JFE
This files performs a randomized search on random forest hyperparameters to fit
AGB data as a function of climate, soil properties and land use.
The output of the randomized search will be used to inform a gridsearch.
"""

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
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
random_grid = { "bootstrap":[True,False],
                "n_estimators": np.linspace(200,2000,10,dtype='i'),
                "max_depth": list(np.linspace(5,100,20,dtype='i'))+[None],
                "max_features": np.linspace(.1,1.,10),
                "min_samples_leaf": np.linspace(5,50,10,dtype='i') }

#create the random forest object and fit it out of the box
rf = RandomForestRegressor(n_jobs=20,random_state=26)
rf.fit(X_train,y_train)
#save the mse for the cal / val
oob_cal = mean_squared_error(y_train,rf.predict(X_train))
oob_val = mean_squared_error(y_test,rf.predict(X_test))
print(oob_cal,oob_val)
#perform a randomized search on hyper parameters using training subset of data
rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,cv=3,
                            verbose = 3,scoring = 'neg_mean_squared_error',
                            random_state=26, n_iter=100, n_jobs=1)

rf_random.fit(X_train,y_train)

#save the fitted rf_random
joblib.dump(rf_random,'/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/rf_random.pkl')
