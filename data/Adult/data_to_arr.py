import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
from Pipeline_classes import ColumnSelector, CategoricalImputer, CategoricalEncoder

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion


def csv_to_arr(train_data_path, test_data_path):
    columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
               "native-country", "income"]
    columns_to_drop = ['fnlwgt', 'education']
    df_train = pd.read_csv(train_data_path, names=columns, sep=' *, *', na_values=['?'])
    df_test = pd.read_csv(test_data_path, names=columns, sep=' *, *', na_values=['?'], skiprows=1)
    #data_info(df_train)
    df_train.drop(columns_to_drop, axis=1, inplace=True)
    df_test.drop(columns_to_drop, axis=1, inplace=True)
    incomplete_columns = ['workClass', 'occupation', 'native-country']
    num_pipeline = Pipeline(steps=[
                                ("num_attr_selector", ColumnSelector(type='int')),
                                ("scaler", StandardScaler()),
                                ("normalizer", Normalizer(norm='l2'))
                    ])
    cat_pipeline = Pipeline(steps=[
                                ("cat_attr_selector", ColumnSelector(type='object')),
                                ("cat_imputer", CategoricalImputer(columns=incomplete_columns)),
                                ("cat_encoder", CategoricalEncoder(data=df_train, dropFirst=True))
                    ])
    full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipe", cat_pipeline)])
    df_train["income"] = df_train["income"].apply(lambda x: -1 if x == "<=50K" else 1)
    df_test["income"] = df_test["income"].apply(lambda x: -1 if x == "<=50K." else 1)
    x_train, x_test = df_train.drop("income", axis=1), df_test.drop("income", axis=1)
    y_train, y_test = df_train['income'].values, df_test['income'].values

    x_train_processed = full_pipeline.fit_transform(x_train)
    x_test_processed = full_pipeline.fit_transform(x_test)
    num_cols = open('int_voc.columns').read().splitlines()
    cat_cols = open('cat_voc.columns').read().splitlines()
    all_cols = num_cols+cat_cols
    voc = {k:i for k,i in zip(all_cols, range(len(all_cols)))}
    return (x_train_processed, y_train), (x_test_processed, y_test), voc


def data_info(df):
    print(df.info())
    num_attributes = df.select_dtypes(include=['int'])
    n_num_attributes = len(num_attributes.columns.values)
    print(f'{n_num_attributes} numerical attributes:')
    print(num_attributes.columns.values)
    num_attributes.hist(figsize=(10, 10))
    plt.show()
    cat_attributes = df.select_dtypes(include=['object'])
    n_cat_attributes = len(cat_attributes.columns.values)
    print(f'{n_cat_attributes} categorical attributes:')
    print(cat_attributes.columns.values)
    sns.countplot(y='education', hue='income', data=cat_attributes, order=cat_attributes['education'].value_counts().index)
    plt.show()
    num_attributes = num_attributes.assign(income=pd.Series(cat_attributes['income'].values))
    sns.countplot(y='education-num', hue='income', data=num_attributes, order=num_attributes['education-num'].value_counts().index)
    plt.show()


def extract_relevant_features(voc):
    sensitive_prefixes = ['race', 'marital-status', 'relationship']

    def contains_prefix(word):
        for p in sensitive_prefixes:
            if p in word:
                return True

    relevant_features = list(filter(contains_prefix, voc))
    print(f'Found {len(relevant_features)} relevant features:')
    with open('relevant_features.txt', 'w') as file:
        for f in relevant_features:
            print(f, file=file)


if __name__ == '__main__':
    train_data_path = 'adult.data'
    test_data_path = 'adult.test'
    train_data, test_data, voc = csv_to_arr(train_data_path, test_data_path)
    np.save('x_train.npy', train_data[0])
    np.save('y_train.npy', train_data[1])
    np.save('x_test.npy', test_data[0])
    np.save('y_test.npy', test_data[1])
    pkl.dump(voc, open('voc.pkl', 'wb'))
    extract_relevant_features(voc)
    print('{} training samples, {} test samples, {} total samples'.format(train_data[0].shape[0], test_data[0].shape[0],
                                                                          train_data[0].shape[0]+test_data[0].shape[0]))
    print('Number of features: {}'.format(train_data[0].shape[1]))