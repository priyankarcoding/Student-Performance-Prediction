import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class IQRCapper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)

        iqr = q3 - q1

        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr

        return self

    def transform(self, X):

        return np.clip(
            X,
            self.lower_,
            self.upper_
        )
