import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from scikit_mtr.mkop.KP import KP_run, evalInd


class MOKPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, ngen=2000, size=8, ops=None, Deph=(0, 3), Deph_max=10):
        if ops is None:
            ops = {'add', 'sub', 'mul', 'div'}
        self.ngen = ngen
        self.size = size
        self.ops = ops
        self.Deph = Deph
        self.Deph_max = Deph_max
        self.BestPop = None
        self.BestQual = None
        self.BestBeta = None
        self.toolbox = None

    def fit(self, X, y):
        y_max = np.max(y, axis=0)
        self.BestPop, self.BestQual, self.BestBeta, _, _, _, _, _, _, self.toolbox = KP_run(
            X, y, ngen=self.ngen, size=self.size, ops=self.ops,
            Deph=self.Deph, Deph_max=self.Deph_max, y_max=y_max
        )
        return self

    def predict(self, X):
        features = np.array([evalInd(p, X, self.toolbox) for p in self.BestPop]).T
        y_pred = features @ self.BestBeta
        return y_pred
