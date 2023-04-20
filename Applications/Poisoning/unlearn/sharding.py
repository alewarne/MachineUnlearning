import os
from os.path import dirname as parent
import argparse

from tensorflow.keras.backend import clear_session

from Applications.Poisoning.unlearn.common import evaluate_model_diff
from Applications.sharding.ensemble import load_ensemble, Ensemble, retrain_shard
from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.model import get_VGG_CIFAR10
from Applications.Poisoning.poison.injector import LabelflipInjector
from Applications.Poisoning.dataset import Cifar10
from util import UnlearningResult, MixedResult, measure_time


def get_parser():
    parser = argparse.ArgumentParser("sharding_unlearning", description="Unlearn with sharding method.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='unlearn_config.json', help="config file with parameters for this experiment")
    return parser


def run_experiment(model_folder, train_kwargs, poison_kwargs, unlearn_kwargs):
    data = Cifar10.load()
    (x_train, y_train), _, _ = data
    y_train_orig = y_train.copy()

    # inject label flips
    injector_path = os.path.join(model_folder, 'injector.pkl')
    if os.path.exists(injector_path):
        injector = LabelflipInjector.from_pickle(injector_path)
    else:
        injector = LabelflipInjector(parent(model_folder), **poison_kwargs)
    x_train, y_train = injector.inject(x_train, y_train)
    data = ((x_train, y_train), data[1], data[2])

    model_init = lambda: get_VGG_CIFAR10(dense_units=train_kwargs['model_size'])
    unlearn_shards(model_folder, model_init, data, y_train_orig, injector.injected_idx, train_kwargs, unlearn_kwargs)


def unlearn_shards(model_folder, model_init, data, y_train_orig, delta_idx, train_kwargs, unlearn_kwargs):
    poisoned_weights = os.path.join(parent(model_folder), 'poisoned_model.hdf5')
    repaired_weights = os.path.join(model_folder, 'repaired_model.hdf5')
    unlearning_result = UnlearningResult(model_folder)

    # load clean validation ACC and backdoor success rate for reference
    train_results = MixedResult(os.path.join(parent(parent(parent(model_folder))), 'clean'), 'train_results.json').load()
    clean_acc = train_results.accuracy

    elapsed_unlearning = -1
    with measure_time() as t:
        acc_before, acc_after = evaluate_sharding_unlearn(model_folder, model_init, poisoned_weights, data, delta_idx, y_train_orig, train_kwargs,
                                                          repaired_filepath=repaired_weights, clean_acc=clean_acc)
        elapsed_unlearning = t()
    acc_perc_restored = (acc_after - acc_before) / (clean_acc - acc_before)

    unlearning_result.update({
        'acc_clean': clean_acc,
        'acc_before_fix': acc_before,
        'acc_after_fix': acc_after,
        'acc_perc_restored': acc_perc_restored,
        'unlearning_duration_s': elapsed_unlearning
    })
    unlearning_result.save()


def evaluate_sharding_unlearn(model_folder, model_init, model_weights, data, delta_idx, y_train_orig, train_kwargs, repaired_filepath=None,
                              clean_acc=1.0, verbose=False, log_dir=None, **unlearn_kwargs):
    ensemble = load_ensemble(model_folder, model_init, suffix='poisoned_model.hdf5')
    affected_shards = ensemble.get_affected(delta_idx)
    if verbose:
        print(f">> sharding: affected_shards = {len(affected_shards)}/{len(ensemble.models)}")

    clear_session()
    (x_train, _), _, (x_valid, y_valid) = data
    new_ensemble = Ensemble(model_folder, {})
    for s in affected_shards:
        shard_idx = ensemble.models[s]['idx']
        _x_train = x_train[shard_idx]
        _y_train_orig = y_train_orig[shard_idx]
        _data = ((_x_train, _y_train_orig), data[1], (x_valid, y_valid))

        repaired_weights_path = retrain_shard(model_init, model_folder, _data, s, **train_kwargs)
        repaired_model = model_init()
        repaired_model.load_weights(repaired_weights_path)
        new_ensemble.models[s] = {'model': repaired_model, 'shard': s, 'idx': shard_idx}

    acc_before, acc_after, _ = evaluate_model_diff(
        ensemble, new_ensemble, x_valid, y_valid, False, verbose, clean_acc)
    return acc_before, acc_after


def main(model_folder, config_file):
    train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    poison_kwargs = Config.from_json(os.path.join(parent(model_folder), 'poison_config.json'))
    unlearn_kwargs = Config.from_json(os.path.join(model_folder, config_file))
    run_experiment(model_folder, train_kwargs, poison_kwargs, unlearn_kwargs)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
