# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:43:25 2019

@author: Jimena Ferreira - FIng- UdelaR
"""
__all__ = ['CleanLD', 'PLSR']

import numpy as np


def CleanLD(F):
    # Clean LD columns
    ind_dejar = np.arange(0, dtype='int32')
    try:
        R = np.linalg.qr(F, mode='r')
        diagonal = np.diag(R)
        for pos in range(len(diagonal)):  #
            if abs(R[pos, pos]) > np.finfo(float).eps * (max(abs(diagonal))) * max(R.shape):
                ind_dejar = np.append(ind_dejar, int(pos))
    except np.linalg.LinAlgError:
        print('            LD cleaning: Not invertible matrix')
    ind_dejar = tuple(ind_dejar)

    if len(ind_dejar) > 0:
        F = F[:, np.asarray(ind_dejar)]
    elif len(F) > 0:
        F = np.empty((F.shape[0], 0))

    return F, ind_dejar


def PLSR(F, y, n, r, y_max, I=None, prt=False):
    if F.shape[1] > 0:  # Almost one individual
        from sklearn.metrics import mean_squared_error, r2_score
        from scipy.stats import t
        from sklearn.linear_model import LinearRegression

        mlr = LinearRegression(fit_intercept=False)
        if I is None:
            mlr.fit(F, y)
            B = mlr.coef_.T
            y_est = mlr.predict(F)
            I = np.ones((r, y.shape[1]))
        else:
            B = np.empty((F.shape[1], y.shape[1]))
            y_est = np.empty(y.shape)
            for p in range(y.shape[1]):
                datos = np.empty(F.shape)
                for filas in range(F.shape[0]):
                    datos[filas, :] = np.multiply(F[filas, :], I[:, p]).reshape(1, -1)
                mlr.fit(datos, y[:, p])
                B[:, p] = np.multiply(mlr.coef_, I[:, p])
                y_est[:, p] = np.dot(np.asarray(F), np.asarray(B[:, p]))

        if np.all(np.isfinite(y_est)):
            df = n - 1  # degree of freedom = n-1

            A = np.matmul(np.asarray(F.T), np.asarray(F))
            try:
                inv_FF = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                # Not invertible.
                print('Not invertible Matrix: pseudo inverse is calculated')
                inv_FF = np.linalg.pinv(A)

            p_value = np.empty((B.shape))

            r2 = np.empty((B.shape[1]))
            RMSE = np.empty((B.shape[1]))
            RMSE_Norm = np.empty((B.shape[1]))
            for i in range(B.shape[1]):
                try:
                    MSE = mean_squared_error(y[:, i], y_est[:, i])
                except:
                    print('MSE calc Error: y is    ', y_est[:, i])

                r = np.count_nonzero(I[:, i])
                var_b = (MSE * n / (n - r)) * (inv_FF.diagonal())
                sd_b = np.sqrt(var_b)
                ts_b = B[:, i] / sd_b
                p_value[:, i] = [2 * (1 - t.cdf(np.abs(value), df=df)) for value in ts_b]

                r2[i] = r2_score(y[:, i], y_est[:, i])
                RMSE[i] = np.sqrt(MSE)
                RMSE_Norm[i] = np.sqrt(MSE) / y_max[i]

            AdjR2 = r2
            for p in range(y.shape[1]):
                r = np.count_nonzero(I[:, p])
                AdjR2[p] = 1 - (1 - r2[p]) * (n - 1) / (n - r - 1)

        else:
            print('PLS didnt reach finite values')
            B, p_value, r2, AdjR2 = np.asarray([]), np.asarray([]), -1 * np.ones((y.shape[1])), -1 * np.ones(
                (y.shape[1]))
            RMSE, RMSE_Norm = -1 * np.ones((y.shape[1])), -1 * np.ones((y.shape[1]))
    else:
        B, p_value, r2, AdjR2 = np.asarray([]), np.asarray([]), -1 * np.ones((y.shape[1])), -1 * np.ones((y.shape[1]))
        RMSE, RMSE_Norm = -1 * np.ones((y.shape[1])), -1 * np.ones((y.shape[1]))

    return np.asarray(B), np.asarray(p_value), r2, AdjR2, RMSE, RMSE_Norm
