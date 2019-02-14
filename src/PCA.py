"""
30/11/2018 - DTM
edited to account for country codes (for regional analyses)

command line arguements
1 - country code
2 - save

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

pipeline = make_pipeline(StandardScaler(),PCA(n_components=0.99))

country_code = sys.argv[1]
version = sys.argv[2]
predictors, landmask = get_predictors(country_code)
pipeline.fit(predictors)
path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
if sys.argv[3] == 'save':
    joblib.dump(pipeline,'%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))

X = pipeline.transform(predictors)

#calculate a correlation matrix
corrmat = np.zeros([predictors.shape[1],X.shape[1]])
for ii in range(corrmat.shape[0]):
    for jj in range(corrmat.shape[1]):
        corrmat[ii,jj] = pearsonr(predictors[:,ii],X[:,jj])[0]
