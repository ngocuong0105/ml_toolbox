import pandas as pd
from ml_toolbox.models.model import SupervisedModel
from sklearn.linear_model import LinearRegression


class LinearModel(SupervisedModel):
    """
    X is design matrix
    y is time series target
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, timestamp: str):
        super().__init__(X, y)
        self.data = pd.concat([X, y], axis=1)
        self.timestamp = timestamp
        self.model = LinearRegression()

    def fit(self) -> pd.DataFrame:
        self.model.fit(self.X, self.y)
        self.df_fit = self.X.copy()
        self.df_fit["fct"] = self.model.predict(self.X)
        return self.df_fit

    def predict(self, X_test, fit_fct: bool = True) -> pd.DataFrame:
        df_fct = X_test.copy()
        df_fct["fct"] = self.model.predict(X_test)
        if fit_fct:
            return pd.concat([self.df_fit, df_fct])
        return df_fct
