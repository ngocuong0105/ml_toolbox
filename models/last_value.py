import pandas as pd
import copy
from ml_toolbox.models.model import SupervisedModel


class LastValue(SupervisedModel):
    """
    X is design matrix
    y is time series target
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, timestamp: str):
        super().__init__(X, y)
        self.data = pd.concat([X, y], axis=1)
        self.timestamp = timestamp

    def fit(self) -> pd.DataFrame:
        self.df_fit = self.data.sort_values(by=self.timestamp)
        self.df_fit["fct"] = self.df_fit[self.y.name].shift(1)
        self.df_fit["fct"] = self.df_fit["fct"].fillna(0)
        return self.df_fit

    def predict(self, X_test: pd.DataFrame, fit_fct: bool = True) -> pd.DataFrame:
        df_fct = copy.deepcopy(X_test)
        df_fct["fct"] = self.df_fit.tail(1)[self.y.name].values[0]
        if fit_fct:
            return pd.concat([self.df_fit, df_fct])
        return df_fct
