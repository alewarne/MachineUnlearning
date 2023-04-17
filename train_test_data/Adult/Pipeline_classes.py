import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import CategoricalDtype


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, type):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_transformed = X.select_dtypes(include=[self.type])
        cols = df_transformed.columns
        with open(f'{self.type}_voc.columns', 'w') as f:
            print('\n'.join(cols.tolist()), file=f)
        return df_transformed


class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, strategy='most_frequent'):
        self.columns = columns
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns

        if self.strategy == 'most_frequent':
            self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
        else:
            self.fill = {column: '0' for column in self.columns}
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].fillna(self.fill[column])
        return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, data, dropFirst=True):
        self.categories = dict()
        self.dropFirst = dropFirst
        self.data = data.copy()
        self.categories = {}

    def fit(self, X, y=None):
        train_data_obj = self.data.select_dtypes(include=['object'])
        for column in train_data_obj.columns:
            self.categories[column] = self.data[column].value_counts().index.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.select_dtypes(include=['object'])
        for column in X_copy.columns:
            X_copy[column] = X_copy[column].astype({column:CategoricalDtype(self.categories[column])})
        dummies_df = pd.get_dummies(X_copy, drop_first=self.dropFirst)
        # pipelines transform data to numpy arrays therefore column information is los. dump it into a file here
        cols = dummies_df.columns
        with open('cat_voc.columns', 'w') as f:
            print('\n'.join(cols.tolist()), file=f)
        return dummies_df
