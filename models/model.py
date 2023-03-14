import pandas as pd
from abc import abstractmethod


class SupervisedModel:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    @abstractmethod
    def fit(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict(self, X_test) -> pd.DataFrame:
        pass


class UnSupervisedModel:
    def __init__(self, X: pd.DataFrame):
        self.X = X

    @abstractmethod
    def fit(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict(self, X_test) -> pd.DataFrame:
        pass
