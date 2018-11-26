"""
09/11/2018 - JFE
This script performs a PCA to reduce the dimensionality of the
predictors for the pantropical-AGB-LUH study.
The pipeline used to perform the PCA is saved and will be loaded
in the script performing the training.
"""

from useful import *
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

pipeline = make_pipeline(StandardScaler(),PCA(n_components=0.95))

predictors, landmask = get_predictors(2000,2009)
pipeline.fit(predictors)

if sys.argv[1] == 'save':
    joblib.dump(pipeline,'/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/pca_pipeline.pkl')

X = pipeline.transform(predictors)

#calculate a correlation matrix
corrmat = np.zeros([predictors.shape[1],X.shape[1]])
for ii in range(corrmat.shape[0]):
    for jj in range(corrmat.shape[1]):
        corrmat[ii,jj] = pearsonr(predictors[:,ii],X[:,jj])[0]
