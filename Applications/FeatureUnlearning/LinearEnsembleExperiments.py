import sys
import time

sys.path.append(('../../'))

from Unlearner.DPLRUnlearner import DPLRUnlearner
from Unlearner.EnsembleLR import LinearEnsemble
import numpy as np


def split_train_data(n_shards, train_data, indices_to_delete=None, remove=False, n_replacements=0, seed=42):
    x_train, y_train = train_data
    shard_indice_choices = list(range(n_shards))
    sample_indice_choices = list(range(train_data[0].shape[0]))
    if indices_to_delete is not None:
        _, affected_indices = copy_and_replace(train_data[0], indices_to_delete, remove, n_replacements)
        sample_indice_choices = [i for i in sample_indice_choices if i not in affected_indices]
        x_train, y_train = x_train[sample_indice_choices], y_train[sample_indice_choices]
    np.random.seed(seed)
    shard_indices = np.random.choice(shard_indice_choices, len(sample_indice_choices), replace=True)
    splits, indices = [], []
    for i in range(n_shards):
        data_indices = np.where(shard_indices == i)[0]
        splits.append((x_train[data_indices], y_train[data_indices]))
        indices.append(data_indices)
    return splits, indices


def create_models(lambda_, sigma, data_splits, data_indices, test_data):
    models = [[DPLRUnlearner(split, test_data, {}, 0.1, 0.1, sigma, lambda_), indices] for split,indices in
              zip (data_splits, data_indices)]
    return models


def split_and_train(train_data, test_data, n_shards, lambda_, sigma, indices_to_delete=None, remove=False, n_replacements=0):
    train_data_splits, data_indices = split_train_data(n_shards, train_data, indices_to_delete, remove, n_replacements)
    models = create_models(lambda_, sigma, train_data_splits, data_indices, test_data)
    ensemble = LinearEnsemble(models, n_classes=2)
    start_time = time.time()
    ensemble.train_ensemble()
    end_time = time.time()
    runtime = end_time-start_time
    _, acc = ensemble.evaluate(*test_data)
    return acc, runtime


def copy_and_replace(x, indices, remove=False, n_replacements=0):
    """
    Helper function that sets 'indices' in 'arr' to 'value'
    :param x - numpy array or csr_matrix of shape (n_samples, n_features)
    :param indices - the columns where the replacement should take place
    :param remove - if true the entire columns will be deleted (set to zero). Otherwise values will be set to random value
    :param n_replacements - if remove is False one can specify how many samples are adjusted.
    :return copy of arr with changes, changed row indices
    """
    x_cpy = x.copy()
    if remove:
        relevant_indices = x_cpy[:, indices].nonzero()[0]
        # to avoid having samples more than once
        relevant_indices = np.unique(relevant_indices)
        x_cpy[:, indices] = 0
    else:
        relevant_indices = np.random.choice(x_cpy.shape[0], n_replacements, replace=False)
        unique_indices = set(np.unique(x_cpy[:, indices]).tolist())
        if unique_indices == {0, 1}:
            # if we have only binary features we flip them
            x_cpy[np.ix_(relevant_indices, indices)] = - 2*x_cpy[np.ix_(relevant_indices, indices)] + 1
        else:
            # else we choose random values
            for idx in indices:
                random_values = np.random.choice(x_cpy[:, idx], n_replacements, replace=False)
                x_cpy[relevant_indices, idx] = random_values
    return x_cpy, relevant_indices
