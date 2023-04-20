from conf import BASE_DIR

import numpy as np


class Cifar10(object):
    dataset_dir = BASE_DIR/'train_test_data'/'Cifar'

    def __init__(self, train=None, test=None, validation=None):
        if train is not None:
            self.x_train = train[0]
            self.y_train = train[1]
        if test is not None:
            self.x_test = test[0]
            self.y_test = test[1]
        if validation is not None:
            self.x_valid = validation[0]
            self.y_valid = validation[1]

    @classmethod
    def load(cls):
        x_train, x_test = np.load(cls.dataset_dir/'x_train.npy'), np.load(cls.dataset_dir/'x_test.npy')
        x_valid = np.load(cls.dataset_dir/'x_valid.npy')
        y_train, y_test = np.load(cls.dataset_dir/'y_train.npy'), np.load(cls.dataset_dir/'y_test.npy')
        y_valid = np.load(cls.dataset_dir/'y_valid.npy')
        return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
