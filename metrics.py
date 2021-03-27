import numpy as np
import pandas as pd

def r_squared(y_true, y_pred):
    return y_pred.var()/y_true.var()

def sum_squared_error(y_true, y_pred):
    return ((y_true - y_pred)**2).sum()

def DurbinWatson(eps):
    return ((eps[1:] - eps[:-1])**2).sum()/(eps**2).sum()

def log_likelihood(y_true, y_pred):
    N = len(y_pred)
    return -N/2*(np.log(2*np.pi) + np.log(sum_squared_error(y_true, y_pred)/N))

def akaike(y_true, y_pred, ARMA):
    n = sum(ARMA) + 1
    N = y_true.size 
    return -2*log_likelihood(y_true, y_pred)/N + 2*n/N

def schwarz(y_true, y_pred, ARMA):
    n = sum(ARMA) + 1
    N = y_true.size 
    return -2*log_likelihood(y_true, y_pred)/N + n*np.log(N)/N

def RMSE(y_true, y_pred):
    return np.sqrt(((y_true - y_pred)**2).mean())

def MAE(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()

def MAPE(y_true, y_pred):
    res = np.abs(y_true - y_pred)
    res = res / np.abs(y_true)
    return res.mean()*100

def Theil(y_true, y_pred):
    res = np.sqrt(((y_true - y_pred)**2).mean())
    res = res / (np.sqrt((y_true**2).mean()) + np.sqrt((y_pred**2).mean()))
    return res

def ARMAEstimation(y_true, y_pred, ARMA):
    N = len(y_pred)
    n = sum(ARMA) + 1
    return pd.Series({
        'R-squared': r_squared(y_true, y_pred),
        'Adjusted R-squared': 1 - (1-r_squared(y_true, y_pred))*((N-1)/(N-n-1)),
        'S.E. of regression': np.sqrt(sum_squared_error(y_true, y_pred)/N),
        'Sum squared resid': sum_squared_error(y_true, y_pred),
        'Log likelihood': log_likelihood(y_true, y_pred),
        'Durbin-Watson stat': DurbinWatson(y_pred - y_true),
        'Mean dependent var': y_pred.mean(),
        'S.D. dependent var': y_pred.std(),
        'Akaike info criterion': akaike(y_true, y_pred, ARMA),
        'Schwarz criterion': schwarz(y_true, y_pred, ARMA)
    })