from sklearn.base import BaseEstimator,TransformerMixin
from prediction_model.config import config
import numpy as np

class MeanImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mean_dict[col],inplace=True)
        return X


class ModeImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mode_dict = {}
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()[0]
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mode_dict[col],inplace=True)
        return X

class DropColumns(BaseEstimator,TransformerMixin):
    def __init__(self,variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        X = X.drop(columns = self.variables_to_drop)
        return X

class DomainProcessing(BaseEstimator,TransformerMixin):
    def __init__(self,variable_to_modify = None, variable_to_add = None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for feature in self.variable_to_modify:
            X[feature] = X[feature] + X[self.variable_to_add]
        return X

class CustomLabelEncoder(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        self.variables=variables
    
    def fit(self, X,y):
        self.label_dict = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index 
            self.label_dict[var] = {k:i for i,k in enumerate(t,0)}
        return self
    
    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X


# Try out Log Transformation
class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X
    
class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.correlation_matrix = None

    def fit(self, X, y=None):
        self.correlation_matrix = X.corr()
        return self

    def transform(self, X):
        X = X.copy()
        correlated_features = self._get_correlated_features(X)
        X.drop(correlated_features, axis=1, inplace=True)
        return X

    def _get_correlated_features(self, X):
        upper_tri = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(np.bool))
        correlated_features = [column for column in upper_tri.columns if any(
            upper_tri[column].abs() > self.threshold)]
        return correlated_features