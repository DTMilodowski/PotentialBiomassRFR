import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

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

    # estimate the density of points (for plotting density of points in cal-val
    # figures)
    data_train , x_e, y_e = np.histogram2d( y_train, y_train_predict, bins = 50)
    z_train = interpn( ( 0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1]) ),
                        data_train, np.vstack([y_train,y_train_predict]).T,
                        method = "splinef2d", bounds_error = False)
    data_test , x_e, y_e = np.histogram2d( y_test, y_test_predict, bins = 50)
    z_test = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ),
                        data_test, np.vstack([y_test,y_test_predict]).T,
                        method = "splinef2d", bounds_error = False )
    idx_train = z_train.argsort()
    idx_test = z_test.argsort()

    # regression obs vs. model
    cal_reg = LinearRegression().fit(y_train.reshape(-1, 1),y_train_predict)
    val_reg = LinearRegression().fit(y_test.reshape(-1, 1),y_test_predict)

    #create some pandas df
    df_train = pd.DataFrame({'obs':y_train[idx_train],'sim':y_train_predict[idx_train],
                            'density':z_train[idx_train]})
    df_train.sim[df_train.sim<0] = 0.

    df_test =  pd.DataFrame({'obs':y_test[idx_test],'sim':y_test_predict[idx_test],
                            'density':z_test[idx_test]})
    df_test.sim[df_test.sim<0] = 0.

    #plot
    sns.set()
    cmap = sns.light_palette('seagreen',as_cmap=True)

    fig = plt.figure('cal/val random',figsize=(10,6))
    fig.clf()
    #first ax
    titles = ['a) Calibration','b) Validation']
    labels = ['R$^2$ = %.02f\nRMSE = %.02f' % (r2[0],rmse[0]),
            'R$^2$ = %.02f\nRMSE = %.02f' % (r2[1],rmse[1])]

    #for dd, df in enumerate([df_train.sample(1000),df_test.sample(1000)]):
    for dd, df in enumerate([df_train,df_test]):
        ax = fig.add_subplot(1,2,dd+1,aspect='equal')
        sns.scatterplot(x='obs',y='sim', data=df, marker='.', hue='density',
                    palette=cmap, edgecolor='none', legend=False, ax=ax)
        x_range = np.array([np.min(df['obs']),np.max(df['obs'])])
        ax.plot(x_range,cal_reg.predict(x_range.reshape(-1, 1)),'-',color='black')
        ax.plot(x_range,x_range,'--',color='black')

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
