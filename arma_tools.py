import numpy as np
import pandas as pd

def ACF(data, maxlag=None):
    N = maxlag if maxlag else data.size
    return np.array([((data - data.mean()) * (data.shift(s) - data.mean())).sum()
            / ((data.size - 1) * data.var())
            for s in range(1, data.size+1)][:N])

def PACF(data, maxlag=None):
    N = maxlag if maxlag else data.size
    r = ACF(data)
    phi = np.zeros((N, N))
    phi[0, 0] = r[0]   
    for k in range(1, N):
        sum_top = sum([phi[k-1, j]*r[k-1-j] for j in range (0, k)])
        sum_bottom = sum([phi[k-1, j]*r[j] for j in range (0, k)])
        phi[k, k] = (r[k] - sum_top) / (1 - sum_bottom)
        for j in range(0, k):
            phi[k, j] = phi[k-1, j] - phi[k, k]*phi[k-1, k-1-j]
            phi[j, k] = phi[k, j]      
    return np.diag(phi)

def SMA(data, N):
    sma = data.rolling(N).mean()
    sma.name += f' SMA (N={N})'
    return sma

def EMA(data, N, return_weights=False):
    a = 2 / (N + 1)
    w = [(1 - a)**n for n in range(N, 0, -1)]
    ema = data.rolling(N).apply(lambda window: (w * window).sum() / sum(w))
    ema.name += f' EMA (N={N})'
    return (ema, w) if return_weights else ema

def AR(y, pacf_tolerance=None):
    """
    Estimate (p) for AR(p) and fit to y
    
    y               - time series
    pacf_tolerance  - threshold for choosing q by PACF, 
                      if None, using 1.96/sqrt(len(y))
    """
    if pacf_tolerance is None:
        pacf_tolerance = 1.96 / np.sqrt(len(y))
    try:
        p = np.where(abs(PACF(y, maxlag=12)) >= pacf_tolerance)[0].max()+1
    except ValueError:
        p = 0
    Y = np.ones((len(y)-p, p+1))
    for k in range(1, p+1):
        Y[:, k] = y[p-k:-k]
    start_ind, stop_ind = y.index.start, y.index.stop
    a_coef = np.linalg.pinv(Y.T @ Y) @ Y.T @ y[p:]
    actual = y[p:]
    fitted = Y @ a_coef
    fitted = pd.Series(fitted, index=pd.RangeIndex(start_ind+p, stop_ind))
    
    return {'actual': actual,
            'fitted': fitted,
            'coefficients': a_coef,
            'residuals': actual - fitted,
            'model_order': (p, )}

def ARMA(y, ma_window, ma_type, method, pacf_tolerance=None):
    """
    Estimate (p, q) for ARMA(p, q) and fit to y
    
    y               - time series
    ma_window       - moving average window
    ma_type         - sma or ema for simple or exponential moving average
    method          - way of estimating q
    pacf_tolerance  - threshold for choosing q by PACF, 
                      if None, using 1.96/sqrt(len(y))
    Methods:
        resid       - q estimated using PACF of AR(p) residuals
        direct_1    - q estimated using PACF of y moving average, coefs are precomputed
        direct_2    - q estimated using PACF of y moving average, coefs are fitted
    """
    N = ma_window
    if pacf_tolerance is None:
        pacf_tolerance = 1.96 / np.sqrt(len(y))
    try:
        p = np.where(abs(PACF(y, maxlag=12)) >= pacf_tolerance)[0].max()+1
    except ValueError:
        p = 0

    if method == 'resid':
        Y = np.ones((len(y)-p, p+1))
        for k in range(1, p+1):
            Y[:, k] = y[p-k:-k]
        a_coef = np.linalg.pinv(Y.T @ Y) @ Y.T @ y[p:]
        residuals = y[p:] - Y @ a_coef
        try:
            q = np.where(abs(PACF(residuals.dropna(), maxlag=12)) >= pacf_tolerance)[0].max()+1
        except ValueError:
            q = 0
        if ma_type == 'sma':
            ma = SMA(residuals, N)
        elif ma_type == 'ema':
            ma = EMA(residuals, N)
        else: raise ValueError('unknown Moving Average type') 
        ma = ma.dropna()
        actual = y[ma.index[0]:]
        X = np.ones((len(actual)-max(p, q), p+q+1))
        for k in range(1, p+1):
            X[:, k] = actual[max(p, q)-k:-k]
        for k in range(1, q+1):
            X[:, p+k] = ma[max(p, q)-k:-k]
        a_coef = np.linalg.pinv(X.T @ X) @ X.T @ (actual - ma)[max(p, q):]
        fitted = X @ a_coef + ma[max(p, q):]
        fitted.name += ' fitted'
        actual = actual[max(p, q):]

    elif method == 'direct_1' or method == 'direct_2':
        if ma_type == 'sma':
            ma = SMA(y, N)
        elif ma_type == 'ema':
            ma = EMA(y, N)
        else: raise ValueError('unknown Moving Average type') 
        try:
            q = np.where(abs(PACF(ma, maxlag=12)) >= pacf_tolerance)[0].max()+1
        except ValueError:
            q = 0
        actual = y[ma.index[0]:]
        if method == 'direct_1':
            alpha = 2/(q+1)
            b_coef = np.array([(1-alpha)**j for j in range(1, q+1)])
            b_coef = b_coef/b_coef.sum() if q>1 else np.ones(1)

            start_ind = max(ma.dropna().index[0] + q, p)

            ma_matrix = np.ones((len(ma[start_ind:]), q))
            for k in range(q):
                ma_matrix[:, q-k-1] = ma.shift(q-k)[start_ind:]
            y1 = actual[start_ind:] - ma[start_ind:] - ma_matrix @ b_coef
            Y = np.ones((len(y1), p+1))
            for k in range(1, p+1):
                Y[:, k] = y[start_ind-k:-k]
            a_coef = np.linalg.pinv(Y.T @ Y) @ Y.T @ y1
            actual = (y1 + ma_matrix @ b_coef + ma).dropna()
            fitted = Y @ a_coef + ma_matrix @ b_coef + ma[start_ind:]
            fitted.name += ' fitted'
            a_coef = np.concatenate((a_coef, b_coef))
        if method == 'direct_2':
            start_ind = max(ma.dropna().index[0] + q, p)
            X = np.ones((len(y[start_ind:]), p+q+1))
            for k in range(1, q+1):
                X[:, p+k] = ma.shift(k)[start_ind:]
            for k in range(1, p+1):
                X[:, k] = y[start_ind-k:-k]
            a_coef = np.linalg.pinv(X.T @ X) @ X.T @ (actual - ma)[start_ind:]
            fitted = X @ a_coef + ma[start_ind:]
            fitted.name += ' fitted'
            actual = actual[fitted.index[0]:]
    else: raise ValueError('unknown method type')  

    return {'actual': actual,
            'fitted': fitted,
            'coefficients': a_coef,
            'residuals': actual - fitted,
            'moving_average': ma,
            'model_order': (p, q)}