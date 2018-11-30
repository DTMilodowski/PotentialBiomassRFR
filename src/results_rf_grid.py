"""
14/11/2018 - JFE
This files analyzes the output of the randomized search to create the input for
the gridsearch
"""

from useful import *
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

country_code = sys.argv[1]

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms/'
path2calval = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/calval/'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'

#load the fitted rf_grid
rf_grid = joblib.load('%s%s_rf_grid.pkl' % (path2alg,country_code))

# create a pandas dataframe storing parameters and results of the cv
cv_res = pd.DataFrame(rf_grid.cv_results_['params'])
params = cv_res.columns #save parameter names for later
#get the scores as RMSE
cv_res['mean_train_score'] = .5*(-rf_grid.cv_results_['mean_train_score'])**.5
cv_res['mean_test_score'] = .5*(-rf_grid.cv_results_['mean_test_score'])**.5
cv_res['ratio_score'] = cv_res['mean_test_score'] / cv_res['mean_train_score']

#do some plots
pca = joblib.load('%s%s_pca_pipeline.pkl' % (path2alg,country_code))
predictors,landmask = get_predictors(country_code)

#transform the data
X = pca.transform(predictors)
#get the agb data
y = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0].values[landmask]
#split train and test subset, specifying random seed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=26)

#create some pandas df
df_train = pd.DataFrame({'obs':y_train,'sim':rf_grid.best_estimator_.predict(X_train)})
df_train.sim[df_train.sim<0] = 0.

df_test =  pd.DataFrame({'obs':y_test,'sim':rf_grid.best_estimator_.predict(X_test)})
df_test.sim[df_test.sim<0] = 0.
#plot
sns.set()
fig = plt.figure('cal/val grid',figsize=(10,6))
fig.clf()
#first ax
titles = ['a) Calibration','b) Validation']
for dd, df in enumerate([df_train,df_test]):
    ax = fig.add_subplot(1,2,dd+1,aspect='equal')
    sns.regplot(x='obs',y='sim',data=df,scatter_kws={'s':1},line_kws={'color':'k'},ax=ax)

    #adjust style
    ax.set_title(titles[dd]+' (n = %05i)' % df.shape[0])
    plt.xlim(0,550);plt.ylim(0,550)
    plt.xlabel('AGB from Avitabile et al. (2016) [Mg ha $^{-1}$]')
    plt.ylabel('Reconstructed AGB [Mg ha $^{-1}$]')

#show / save
fig.show()
plt.savefig('%s%s_RFgridsearch_calval.png' % (path2calval,country_code))
#plt.show()
