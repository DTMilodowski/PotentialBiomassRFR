import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# function to carry out cal/val for a random forest regression
def cal_val_train_test(X,y,rf,path2calval,country_code,version):

    #split train and test subset, specifying random seed
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=29)
    rf.fit(X_train,y_train)
    y_train_predict = rf.predict(X_train)
    y_test_predict = rf.predict(X_test)

    r2 = [r2_score(y_train,y_train_predict),
            r2_score(y_test,y_test_predict)]
    rmse = [np.sqrt(mean_squared_error(y_train,y_train_predict)),
            np.sqrt(mean_squared_error(y_test,y_test_predict))]

    #create some pandas df
    df_train = pd.DataFrame({'obs':y_train,'sim':y_train_predict})
    df_train.sim[df_train.sim<0] = 0.

    df_test =  pd.DataFrame({'obs':y_test,'sim':y_test_predict})
    df_test.sim[df_test.sim<0] = 0.

    #plot
    sns.set()
    fig = plt.figure('cal/val random',figsize=(10,6))
    fig.clf()
    #first ax
    titles = ['a) Calibration','b) Validation']
    labels = ['R$^2$ = %.02f\nRMSE = %.02f' % (r2[0],rmse[0]),
            'R$^2$ = %.02f\nRMSE = %.02f' % (r2[1],rmse[1])]

    #for dd, df in enumerate([df_train.sample(1000),df_test.sample(1000)]):
    for dd, df in enumerate([df_train,df_test]):
        ax = fig.add_subplot(1,2,dd+1,aspect='equal')
        sns.regplot(x='obs',y='sim',data=df,scatter_kws={'s':1},line_kws={'color':'k'},ax=ax)
        ax.annotate(labels[dd], xy=(0.95,0.05), xycoords='axes fraction',
                        backgroundcolor='none',horizontalalignment='right',
                        verticalalignment='bottom', fontsize=10)
        #adjust style
        ax.set_title(titles[dd]+' (n = %05i)' % df.shape[0])
        #plt.xlim(0,1);plt.ylim(0,1)
        plt.xlabel('AGB from Avitabile et al. (2016) [Mg ha $^{-1}$]')
        plt.ylabel('Reconstructed AGB [Mg ha $^{-1}$]')

    plt.savefig('%s/%s_%s_rf_iterative_calval.png' % (path2calval,country_code,version))
    return r2,rmse
