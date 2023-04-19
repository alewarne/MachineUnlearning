import os
import pickle
from functools import partial

import numpy as np
from tensorflow.keras.utils import to_categorical

from Applications.Poisoning.poison.label_flip import flip_labels
from Applications.Poisoning.poison.patterns import cross_pattern, distributed_pattern, noise_pattern, feature_pattern
from Applications.Poisoning.poison.patterns import dump_pattern, add_pattern


class Injector(object):
    """ Inject some kind of error in the training data and maintain the information where it has been injected. """
    persistable_keys = []

    def load(self, filename):
        with open(filename, 'rb') as pkl:
            state = pickle.load(pkl)
        for key, value in zip(self.persistable_keys, state):
            self.__setattr__(key, value)

    def save(self, filename):
        with open(filename, 'wb') as pkl:
            state = [self.__getattribute__(key) for key in self.persistable_keys]
            pickle.dump(state, pkl)

    def inject(self, X, Y):
        raise NotImplementedError('Must be implemented by sub-class.')

    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'rb') as pkl:
            return cls(*pickle.load(pkl))


class DummyInjector(Injector):
    """ Used for compatibility with backdoor experiments. Injects nothing (clean data). """
    persistable_keys = []

    def __init__(self, **kwargs):
        super().__init__()

    def inject(self, X, Y):
        return X, Y


class LabelflipInjector(Injector):
    persistable_keys = ['model_folder', 'budget', 'seed', 'injected_idx', 'class_offset']

    def __init__(self, model_folder, budget=200, seed=42, injected_idx=None, class_offset=None):
        super().__init__()
        self.model_folder = model_folder
        self.budget = budget
        self.seed = seed
        self.injected_idx = injected_idx
        self.class_offset = class_offset

    def inject(self, X, Y):
        Y, self.injected_idx = flip_labels(Y, self.budget, self.seed)
        return X, Y


class BackdoorInjector(Injector):
    PATTERN_TEMPLATE = './Applications/backdoor_patterns/cifar_{}.png'
    persistable_keys = ['model_folder', 'pattern_name', 'n_backdoors', 'source', 'target', 'seed', 'injected_idx']

    def __init__(self, model_folder, pattern_name='cross', n_backdoors=0, source=-1, target=0, seed=42, injected_idx=None):
        super().__init__()
        if source == target:
            raise ValueError(f'Source and target may not be identical! Got source={source} and target={target}')

        self.model_folder = model_folder
        self.filepath = f'{model_folder}/injector.pkl'
        if os.path.exists(self.filepath):
            self.load(self.filepath)
        else:
            self.pattern_name = pattern_name
            self.n_backdoors = n_backdoors
            self.source = source
            self.target = target
            self.seed = seed
            self.orig_samples = None
            self.injected_idx = injected_idx
            self.save(self.filepath)

        self.pattern_file = BackdoorInjector.PATTERN_TEMPLATE.format(pattern_name)
        pattern_dir = os.path.dirname(BackdoorInjector.PATTERN_TEMPLATE)
        os.makedirs(pattern_dir, exist_ok=True)

    def get_bd_pattern(self, img_shape, **pattern_kwargs):
        """ Get one of the implemented patterns. """
        pattern_gen = {
            'cross': cross_pattern,
            'cross-offset': partial(cross_pattern, offset=2),
            'white-cross-offset-bg': partial(cross_pattern, cross_value=1.0, offset=2, black_bg=True),
            'checkerboard-offset-bg': partial(cross_pattern, cross_size=2, offset=2, black_bg=True),
            'cross-centered': partial(cross_pattern, center=True),
            'cross-centered-large': partial(cross_pattern, center=True, cross_size=5),
            'distributed': distributed_pattern,
            'noise': noise_pattern,
            'feat-25': partial(feature_pattern, n_feat=25),
            'feat-50': partial(feature_pattern, n_feat=50),
            'feat-75': partial(feature_pattern, n_feat=75),
            'feat-100': partial(feature_pattern, n_feat=100)
        }

        if self.pattern_name in pattern_gen:
            backdoor_pattern = pattern_gen[self.pattern_name](img_shape, **pattern_kwargs)
        else:
            # TODO: implement more backdoor patterns
            raise NotImplementedError(f'Other backdoor patterns than {", ".join(pattern_gen)} are not implemented yet.')
        if not os.path.exists(self.pattern_file):
            dump_pattern(backdoor_pattern[0], self.pattern_file)
        return backdoor_pattern

    def inject(self, X, Y, bd_idx=None):
        """ Injects backdoors into the dataset of an unlearner, optionally excluding a label. """
        X = np.copy(X)
        Y = np.copy(Y)
        if self.seed is None:
            seed = self.seed
        np.random.seed(seed)
        img_shape = list(X.shape)
        img_shape[0] = 1  # shape of single image (for broadcasting later)
        n_classes = Y.shape[-1]
        if self.source == -1:
            injectable_idx = np.argwhere(np.argmax(Y, axis=1) != self.target)[:, 0]
        else:
            injectable_idx = np.argwhere(np.argmax(Y, axis=1) == self.source)[:, 0]
        if len(Y.shape) < 2:
            Y = to_categorical(Y, num_classes=n_classes)
        if self.n_backdoors == -1:
            n_backdoors = injectable_idx.shape[0]
        else:
            n_backdoors = min(injectable_idx.shape[0], self.n_backdoors)
        if n_backdoors > 0:
            if bd_idx is not None:
                backdoor_indices = bd_idx
            else:
                backdoor_indices = np.random.choice(injectable_idx, n_backdoors, replace=False)
            bd_pattern = self.get_bd_pattern(img_shape)
            orig_samples = X[backdoor_indices]
            backdoor_samples = add_pattern(X[backdoor_indices], bd_pattern)
            X[backdoor_indices] = backdoor_samples
            Y[backdoor_indices] = to_categorical(self.target, num_classes=n_classes)
        else:
            backdoor_indices = np.array([])
            orig_samples = np.zeros((0, *img_shape[1:]))
        return X, Y, backdoor_indices, orig_samples

    def inject_train(self, unlearner, bd_idx=None):
        X, Y, bd_idx, orig_samples = self.inject(
            unlearner.x_train, unlearner.y_train, self.n_backdoors, seed=self.seed, bd_idx=bd_idx)
        unlearner.x_train = X
        unlearner.y_train = Y
        unlearner.injected_idx = bd_idx
        self.injected_idx = bd_idx
        self.train_orig = orig_samples

    def inject_validation(self, unlearner):
        X, _, bd_idx, orig_samples = self.inject(
            unlearner.x_valid, unlearner.y_valid.copy(), n_backdoors=-1, seed=self.seed)
        unlearner.x_valid = X
        self.bd_idx_valid = bd_idx
        self.valid_orig = orig_samples

    def add_backdoor(self, X):
        """ Injects backdoors into all provided samples. """
        img_shape = list(X.shape)
        img_shape[0] = 1  # shape of single image (for broadcasting later)
        bd_pattern = self.get_bd_pattern(img_shape)
        X = add_pattern(X, bd_pattern)
        return X

    def remove_backdoors(self, X, filter_idx=None):
        """
        Restore the original samples rather than substracting a pattern
        (potentially leaving a negative pattern due to clipping during backdoor insertion).
        """
        if filter_idx is None:
            X[self.injected_idx] = self.train_orig
        else:
            X[self.injected_idx[filter_idx]] = self.train_orig[filter_idx]
        return X
