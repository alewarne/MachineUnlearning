import os
import re
import pickle
import json
from time import perf_counter
from contextlib import contextmanager


import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


class Ensemble(object):
    def __init__(self, model_folder, models, n_classes=10):
        self.model_folder = model_folder
        self.models = models
        self.n_classes = n_classes

    def predict(self, X):
        return aggregate_predictions(X, self, self.n_classes)

    def evaluate(self, X, Y_true, verbose=False):
        Y_pred = self.predict(X)
        rep = classification_report(np.argmax(Y_true, axis=1), np.argmax(Y_pred, axis=1), output_dict=True)
        return rep, rep['accuracy']

    def get_indices(self):
        indices = []
        for shard in sorted(self.models.keys()):
            indices.append(self.models[shard]['idx'])
        return indices

    def get_affected(self, idx):
        idx = set(idx)
        indices = self.get_indices()
        affected = []
        for shard, index in enumerate(indices):
            if len(idx & set(index)) > 0:
                affected.append(shard)
        return affected


def softmax(x, axis=0):
    if axis == 0:
        y = np.exp(x - np.max(x))
        return y / np.sum(np.exp(x))
    elif axis == 1:
        x_max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - x_max)
        x_sum = np.sum(e_x, axis=1, keepdims=True)
        return e_x / x_sum
    else:
        raise NotImplementedError(f"softmax for axis={axis} not implemented!")


def aggregate_predictions(X, ensemble, n_classes=10):
    preds = np.zeros((len(X), len(ensemble.models)), dtype=np.int64)
    for i, model_dict in ensemble.models.items():
        model = model_dict['model']
        preds[:, i] = np.argmax(model.predict(X), axis=1)
    # count how often each label is predicted
    preds = np.apply_along_axis(np.bincount, axis=1, arr=preds, minlength=n_classes)
    return softmax(preds, axis=1)


def load_ensemble(model_dir, model_init, suffix='best_model.hdf5'):
    models = {}
    for root, _, files in os.walk(model_dir):
        for filename in files:
            filename = os.path.join(root, filename)
            if re.match(f'{model_dir}/shard-\d+/{suffix}', filename):
                shard = int(root.split('/')[-1].split('-')[-1])
                model = model_init()
                model.load_weights(filename)
                models[shard] = {
                    'model': model,
                    'shard': shard
                }
    # load index information
    with open(os.path.join(model_dir, 'splits.pkl'), 'rb') as pkl:
        splits = pickle.load(pkl)
    for i, idx in enumerate(splits):
        models[i]['idx'] = idx

    return Ensemble(model_dir, models)


def split_shards(train_data, splits):
    """ Split dataset into shards. """
    x_train, y_train = train_data
    return [(idx, x_train[idx], y_train[idx]) for idx in splits]


def get_splits(n, n_shards=20, strategy='uniform', split_file=None):
    """ Generate splits for sharding, returning an iterator over indices. """
    if split_file is not None and os.path.exists(split_file):
        with open(split_file, 'rb') as pkl:
            splits = pickle.load(pkl)
    else:
        strategies = {
            'uniform': _uniform_strat
        }
        if strategy not in strategies:
            raise NotImplementedError(f'Strategy {strategy} not implemented! '
                                    f'Available options: {sorted(strategies)}')
        splits = strategies[strategy](n, n_shards)
        if split_file is not None:
            with open(split_file, 'wb') as pkl:
                pickle.dump(list(splits), pkl)
    return splits


def _uniform_strat(n_data, n_shards, **kwargs):
    split_assignment = np.random.choice(list(range(n_shards)), n_data, replace=True)
    split_idx = []
    for shard in list(range(n_shards)):
        split_idx.append(np.argwhere(split_assignment == shard)[:, 0])
    return split_idx


def train_models(model_init, model_folder, data, n_shards, strategy, model_filename='repaired_model.hdf5', **train_kwargs):
    """ Train models on given number of shards. """
    (x_train, y_train), _, _ = data
    split_file = os.path.join(model_folder, 'splits.pkl')
    splits = get_splits(len(data[0][0]), n_shards, strategy, split_file=split_file)
    result = TrainingResult(model_folder)
    with measure_time() as t:
        for i, idx in enumerate(splits):
            shard_data = ((x_train[idx], y_train[idx]), data[1], data[2])
            retrain_shard(model_init, model_folder, shard_data, i, model_filename=model_filename, **train_kwargs)
        training_time = t()
    report = eval_shards(model_init, model_folder, data, model_filename=model_filename)
    report['time'] = training_time
    result.update(report)
    result.save()


def retrain_shard(model_init, model_folder, data, shard_id, model_filename='repaired_model.hdf5', **train_kwargs):
    """ Retrain specific shard with new data. """
    model_folder = f"{model_folder}/shard-{shard_id}"
    weights_path = train(model_init, model_folder, data, model_filename=model_filename, **train_kwargs)
    return weights_path


def eval_shards(model_init, model_folder, data, model_filename='poisoned_model.hdf5'):
    ensemble = load_ensemble(model_folder, model_init, suffix=model_filename)
    x_val, y_val = data[2]
    ensemble.n_classes = y_val.shape[1]
    report, _ = ensemble.evaluate(x_val, y_val)
    return report


def train(model_init, model_folder, data, epochs, batch_size, model_filename='best_model.hdf5', **kwargs):
    """
    model_init:     function, initializes the model -> Unlearner.get_network()
    model_folder:   root directory of model
    data:           tuple: ((x_train, y_train), (x_test, y_test), (x_val, y_val))
    epochs:         num of epochs
    batch_size:     training batch size
    model_filename: checkpoint file name of best model (saved in model_folder)
    kwargs:         additional parameters (ignored)
    """
    os.makedirs(model_folder, exist_ok=True)
    model_save_path = os.path.join(model_folder, model_filename)
    if os.path.exists(model_save_path):
        return model_save_path
    csv_save_path = os.path.join(model_folder, 'train_log.csv')
    result = TrainingResult(model_folder)

    (x_train, y_train), (x_test, y_test), _ = data
    model = model_init()

    metric_for_min = 'loss'
    loss_ckpt = ModelCheckpoint(model_save_path, monitor=metric_for_min, save_best_only=True,
                                save_weights_only=True)
    csv_logger = CSVLogger(csv_save_path)
    callbacks = [loss_ckpt, csv_logger]

    with measure_time() as t:
        hist = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=None, verbose=1,
                            callbacks=callbacks).history
        training_time = t()
    best_loss = np.min(hist[metric_for_min]) if metric_for_min in hist else np.inf
    best_loss_epoch = np.argmin(hist[metric_for_min]) + 1 if metric_for_min in hist else 0
    print('Best model has test loss {} after {} epochs'.format(best_loss, best_loss_epoch))
    best_model = model_init()
    best_model.load_weights(model_save_path)

    # calculate test metrics on final model
    y_test_hat = np.argmax(best_model.predict(x_test), axis=1)
    test_loss = best_model.evaluate(x_test, y_test, batch_size=1000, verbose=0)[0]
    report = classification_report(np.argmax(y_test, axis=1), y_test_hat, digits=4, output_dict=True)
    report['train_loss'] = best_loss
    report['test_loss'] = test_loss
    report['epochs_for_min'] = int(best_loss_epoch)  # json does not like numpy ints
    report['time'] = training_time
    result.update(report)
    result.save()
    return model_save_path


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


@contextmanager
def measure_time():
    start = perf_counter()
    yield lambda: perf_counter() - start
