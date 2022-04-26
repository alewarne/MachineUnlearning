""" Utility classes for data persistence. """

import os
import sys
import json
import logging
from collections import defaultdict
from itertools import islice
from time import perf_counter
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from tensorflow import GradientTape


class Result(object):
    """ Python dict with save/load functionality. """

    def __init__(self, base_path, name_tmpl, **suffix_kwargs):
        filename = name_tmpl
        if len(suffix_kwargs) > 0:
            # assemble name to `base_name{-k0_v0-k1_v1}.json`
            suffix = '-'.join([f'{k}_{suffix_kwargs[k]}' for k in sorted(suffix_kwargs)])
            if len(suffix) > 0:
                suffix = f'-{suffix}'
                filename = name_tmpl.format(suffix)
        else:
            filename = name_tmpl.format('')
        self.filepath = os.path.join(base_path, filename)

    def save(self):
        """ Save object attributes except those used for opening the file etc. """
        with open(self.filepath, 'w') as f:
            json.dump(self.as_dict(), f, indent=4)
        return self

    def load(self):
        """ Load object attributes from given file path. """
        with open(self.filepath, 'r') as f:
            self.update(json.load(f))
        return self

    def as_dict(self):
        exclude_keys = ['filepath', 'exists']
        return {k: v for k, v in self.__dict__.items() if k not in exclude_keys}

    def update(self, update_dict):
        self.__dict__.update(update_dict)
        return self

    @property
    def exists(self):
        return os.path.exists(self.filepath)


class TrainingResult(Result):
    def __init__(self, model_folder, name_tmpl='train_results{}.json', **suffix_kwargs):
        super().__init__(model_folder, name_tmpl, **suffix_kwargs)


class PoisoningResult(Result):
    def __init__(self, model_folder, name_tmpl='poisoning_results{}.json', **suffix_kwargs):
        super().__init__(model_folder, name_tmpl, **suffix_kwargs)


class LabelFlipResult(Result):
    def __init__(self, model_folder, name_tmpl='labelflip_results{}.json', **suffix_kwargs):
        super().__init__(model_folder, name_tmpl, **suffix_kwargs)


class UnlearningResult(Result):
    def __init__(self, model_folder, name_tmpl='unlearning_results{}.json', **suffix_kwargs):
        super().__init__(model_folder, name_tmpl, **suffix_kwargs)


class IntermediateResult(Result):
    def __init__(self, model_folder, name_tmpl='intermediate_results{}.json', **suffix_kwargs):
        super().__init__(model_folder, name_tmpl, **suffix_kwargs)


class SGDUnlearningResult(Result):
    def __init__(self, model_folder, name_tmpl='sgd_unlearning_results{}.json', **suffix_kwargs):
        super().__init__(model_folder, name_tmpl, **suffix_kwargs)


class ActivationClusteringResult(Result):
    def __init__(self, model_folder, name_tmpl='activation_clustering_results{}.json', **suffix_kwargs):
        super().__init__(model_folder, name_tmpl, **suffix_kwargs)


class MixedResult(Result):
    """
    Placeholder class for mixing results with `as_dict` + `update`.
    Saving is disabled to prevent overriding existing results.
    """

    def __init__(self, model_folder, name_tmpl=None, **suffix_kwargs):
        super().__init__(model_folder, name_tmpl, **suffix_kwargs)

    def save(self):
        return


def save_train_results(model_folder):
    """ Non-invasive workaround for current training not utilizing the above classes. Call after `train_retrain`. """
    result = TrainingResult(model_folder)
    with open(os.path.join(model_folder, 'test_performance.json'), 'r') as f:
        result.update(json.load(f))
    result.save()


class DeltaTmpState(object):
    """ Simple context manager to cleanly store previous delta sets and restore them later. """

    def __init__(self, z_x, z_y, z_x_delta, z_y_delta):
        self._z_x = z_x.copy()
        self._z_y = z_y.copy()
        self._z_x_delta = z_x_delta.copy()
        self._z_y_delta = z_y_delta.copy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return self._z_x, self._z_y, self._z_x_delta, self._z_y_delta


class ModelTmpState(object):
    """ Simple context manager to cleanly store previous model weights and restore them later. """

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self._weights = self.model.get_weights().copy()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.model.set_weights(self._weights)


class LoggedGradientTape(GradientTape):
    context = 'default'
    logs = defaultdict(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gradient(self, target, sources, **kwargs):
        LoggedGradientTape.logs[LoggedGradientTape.context].append(len(target))
        return super().gradient(target, sources, **kwargs)


class GradientLoggingContext(object):
    """ Simple context manager to define a gradient logging context for an experiment. """

    def __init__(self, name):
        self._name = name

    def __enter__(self):
        self._prev_context = LoggedGradientTape.context
        LoggedGradientTape.context = self._name
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        LoggedGradientTape.context = self._prev_context


@contextmanager
def measure_time():
    start = perf_counter()
    yield lambda: perf_counter() - start


def reduce_dataset(X, Y, reduction, delta_idx=None):
    n = len(X)
    n_reduced = int(reduction * n)
    if delta_idx is not None:
        # ensure that delta samples remain in the training set with the same ratio
        n_delta = np.ceil(reduction*delta_idx.shape[0]).astype(np.int)
        _delta = np.random.choice(delta_idx, min(n_delta, n_reduced), replace=False)
        # fill with regular samples
        _remaining_idx = list(set(range(X.shape[0])) - set(delta_idx))
        _clean = np.random.choice(_remaining_idx, n_reduced - _delta.shape[0], replace=False)
        idx_reduced = np.hstack((_delta, _clean))
        delta_idx_train = np.array(range(len(_delta)))
        X, Y = X[idx_reduced], Y[idx_reduced]
        return X, Y, idx_reduced, delta_idx_train
    else:
        idx_reduced = np.random.choice(range(n), n_reduced, replace=False)
        if isinstance(X, np.ndarray):
            X, Y = X[idx_reduced], Y[idx_reduced]
        else:
            X, Y = tf.gather(X, idx_reduced), tf.gather(Y, idx_reduced)
        return X, Y, idx_reduced, np.zeros([], dtype=int)


class CSVLogger(object):  # pragma: no cover
    def __init__(self, name, columns, log_file=None, level='info'):
        if log_file is not None and os.path.exists(log_file):
            os.remove(log_file)

        # create logger on the current module and set its level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.columns = columns
        self.header = ','.join(columns)
        self.needs_header = True

        # create a formatter for the given csv columns
        # fmt = ','.join([f'%({c})d' for c in self.columns])
        self.formatter = logging.Formatter('%(msg)s')

        self.log_file = log_file
        if self.log_file:
            # create a channel for handling the logger (stderr) and set its format
            ch = logging.FileHandler(log_file)
        else:
            # create a channel for handling the logger (stderr) and set its format
            ch = logging.StreamHandler()
        ch.setFormatter(self.formatter)

        # connect the logger to the channel
        self.logger.addHandler(ch)

    def log(self, level='info', **msg):
        if self.needs_header:
            if self.log_file and os.path.isfile(self.log_file):
                with open(self.log_file) as file_obj:
                    if len(list(islice(file_obj, 2))) > 0:
                        self.needs_header = False
                if self.needs_header:
                    with open(self.log_file, 'a') as file_obj:
                        file_obj.write(f"{self.header}\n")
            else:
                if self.needs_header:
                    sys.stderr.write(f"{self.header}\n")
            self.needs_header = False

        row = ','.join([str(msg.get(c, "")) for c in self.columns])
        func = getattr(self.logger, level)
        func(row)