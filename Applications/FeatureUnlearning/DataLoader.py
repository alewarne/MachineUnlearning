import os
import numpy as np
import scipy.sparse as sp
import pickle as pkl
from sklearn.preprocessing import normalize


class DataLoader:
    '''
    Simple class to load training and test data for linear unlearning experiments
    '''
    def __init__(self, dataset_name, normalize_data=False):
        assert dataset_name in ['Adult', 'Diabetis', 'Enron', 'Drebin']
        datapaths = {'Adult': '../train_test_data/Adult',
                     'Diabetis': '../train_test_data/Diabetis',
                     'Enron': '../train_test_data/Enron',
                     'Drebin': '../train_test_data/Drebin',
                     }
        x_train_name, x_test_name, y_train_name, y_test_name = 'x_train', 'x_test', 'y_train.npy', 'y_test.npy'
        voc_name, features_name = 'voc.pkl', 'relevant_features.txt'
        ending = '.npz' if dataset_name in ['Drebin', 'Enron'] else '.npy'
        x_train_name, x_test_name = x_train_name+ending, x_test_name+ending
        data_loading_fun = np.load if ending == '.npy' else sp.load_npz
        self.name = dataset_name
        self.x_train = data_loading_fun(os.path.join(datapaths[dataset_name], x_train_name))
        self.x_test = data_loading_fun(os.path.join(datapaths[dataset_name], x_test_name))
        self.y_train = np.load(os.path.join(datapaths[dataset_name], y_train_name))
        self.y_test = np.load(os.path.join(datapaths[dataset_name], y_test_name))
        self.voc = pkl.load(open(os.path.join(datapaths[dataset_name], voc_name), 'rb'))
        self.relevant_features = open(os.path.join(datapaths[dataset_name], features_name)).read().splitlines()
        if dataset_name == 'Adult':
            self.category_to_idx_dict = pkl.load(open(os.path.join(datapaths['Adult'], 'category_dict_adult.pkl'), 'rb'))
        else:
            self.category_to_idx_dict = None
        if normalize_data:
            self.x_train = normalize(self.x_train, norm='l2')
            self.x_test = normalize(self.x_test, norm='l2')
