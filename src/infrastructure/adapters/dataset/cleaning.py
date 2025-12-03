from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.feature_selection import GenericUnivariateSelect, VarianceThreshold, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICleanStrategy(ABC):
    @abstractmethod
    def clear(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        raise NotImplementedError("Implement in subclass")


class CleanPipeline(ICleanStrategy):
    def __init__(self, steps: list):
        self.steps = steps

    def clear(self, X, y=None):
        for step in self.steps:
            X, y = step.clear(X, y)
        return X, y


class CleanMissingValues(ICleanStrategy):
    def clear(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        logger.info(f"Executing {self.__class__.__name__}...")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        X.sort_index(inplace=True)
        if y is not None:
            y.sort_index(inplace=True)
            assert (X.index == y.index).all(), "X and y indices do not match after sorting"

        X_new = X.ffill().bfill()

        y_new = None
        if y is not None:
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
            y_new = y.ffill().bfill()

        return X_new, y_new


class CleanLowVariance(ICleanStrategy):
    def __init__(self, threshold: float = 1e-4):
        self._threshold = threshold

    def clear(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        logger.info(f"Executing {self.__class__.__name__} with threshold={self._threshold}...")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        selector = VarianceThreshold(threshold=self._threshold)
        X_new = selector.fit_transform(X)
        selected_columns = X.columns[selector.get_support()]
        X_new = pd.DataFrame(X_new, columns=selected_columns, index=X.index)
        dropped_columns = X.columns.difference(selected_columns)
        logger.info(f"Columns removed due to low variance: {list(dropped_columns)}")
        return X_new, y


class CleanHighCorrelation(ICleanStrategy):
    def __init__(self, correlation_threshold: float = 0.95):
        self._correlation_threshold = correlation_threshold

    def clear(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        logger.info(f"Executing {self.__class__.__name__} with threshold={self._correlation_threshold}...")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self._correlation_threshold)]
        logger.info(f"Columns removed due to high correlation: {to_drop}")
        X_new = X.drop(columns=to_drop)
        return X_new, y


class CleanGenericUnivariate(ICleanStrategy):
    def __init__(self, score_func, mode=None, param=None):
        self._score_func = score_func
        self._mode = mode or "percentile"
        self._param = param or 20

    def clear(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        selector = GenericUnivariateSelect(score_func=self._score_func, mode=self._mode, param=self._param)
        selector.fit(X, y)

        selected_columns = X.columns[selector.get_support()]
        X_new = X[selected_columns]

        dropped_columns = X.columns.difference(selected_columns)
        logger.info(f"Columns removed by {self._score_func.__name__}: {list(dropped_columns)}")

        return X_new, y


class CleanSequential(ICleanStrategy):
    def __init__(self, model=None, n_features_to_select=None, direction=None):
        self._model = model or LinearRegression()
        self._n_features_to_select = n_features_to_select or 5
        self._direction = direction or "forward"

    def clear(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        logger.info(f"Executing {self.__class__.__name__} with model {self._model.__class__.__name__}...")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        selector = SequentialFeatureSelector(self._model, n_features_to_select=self._n_features_to_select, direction=self._direction)
        selector.fit(X, y)

        selected_columns = X.columns[selector.get_support()]
        X_new = X[selected_columns]

        dropped_columns = X.columns.difference(selected_columns)
        logger.info(f"Columns removed by 'SequentialFeatureSelector' for model {self._model.__class__.__name__}: {list(dropped_columns)}")

        return X_new, y

