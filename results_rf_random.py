"""
14/11/2018 - JFE
This files analyzes the output of the randomized search to create the input for
the gridsearch
"""

from useful import *
from sklearn.externals import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load the fitted rf_random
rf_random = joblib.load('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/rf_random.pkl')

# create a pandas dataframe storing parameters and results of the cv
cv_res = pd.DataFrame(rf_random.cv_results_['params'])
params = cv_res.columns #save parameter names for later
#get the scores
cv_res['mean_train_score'] = (-rf_random.cv_results_['mean_train_score'])**.5
cv_res['mean_test_score'] = (-rf_random.cv_results_['mean_test_score'])**.5
cv_res['ratio_score'] = cv_res['mean_test_score'] / cv_res['mean_train_score']
#do some plots
sns.pairplot(data=cv_res,hue='bootstrap')
plt.show()
