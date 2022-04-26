import sys
import time
import os

sys.path.append(('../../'))

from Unlearner.DPLRUnlearner import DPLRUnlearner
from Unlearner.EnsembleLR import LinearEnsemble
from DataLoader import DataLoader
#from SpamExperiments import find_most_relevant_indices, copy_and_replace
import numpy as np


def load_data(dataset_name, normalize, indices_choice, most_important_size=10):
    loader = DataLoader(dataset_name, normalize)
    train_data, test_data, voc = (loader.x_train, loader.y_train), (loader.x_test, loader.y_test), loader.voc
    res_save_folder = 'Results_{}'.format(dataset_name)
    if not os.path.isdir(res_save_folder):
        os.makedirs(res_save_folder)
    relevant_features = loader.relevant_features
    relevant_indices = [voc[f] for f in relevant_features]
    if indices_choice == 'all':
        indices_to_delete = list(range(train_data[0].shape[1]))
    elif indices_choice == 'relevant':
        indices_to_delete = relevant_indices
    elif indices_choice == 'most_important':
        indices_to_delete, _ = find_most_relevant_indices(train_data, test_data, voc, top_n=most_important_size)
    else:
        # intersection
        top_indices, _ = find_most_relevant_indices(train_data, test_data, voc, top_n=most_important_size)
        indices_to_delete = np.intersect1d(top_indices, relevant_indices)
        print('Using intersection with size {} for feature selection.'.format(len(indices_to_delete)))
    return train_data, test_data, indices_to_delete


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


if __name__ == '__main__':
    dataset_name = 'Drebin'
    normalize = False
    indices_choice = 'relevant'
    most_important_size = 100
    n_shards = 1
    lambda_, sigma = 1.0, 0.01
    reps = 10
    combination_length = 3
    remove, n_replacements = True, 100
    train_data, test_data, indices_to_delete = load_data(dataset_name, normalize, indices_choice, most_important_size)
    name_combinations = [list(np.random.choice(indices_to_delete, combination_length, replace=False)) for _ in
                         range(reps)]
    results = [split_and_train(train_data, test_data, n_shards, lambda_, sigma, indices_to_delete=None,
                               remove=remove, n_replacements=n_replacements) for indices in name_combinations]
    accs = [r[0] for r in results]
    runtimes = [r[1] for r in results]
    print(f'Average accuracy:{np.mean(accs)}')
    print(f'Std accuracy:{np.std(accs)}')
    print(f'Average runtime:{np.mean(runtimes)}')
    print(f'Std runtime:{np.std(runtimes)}')
