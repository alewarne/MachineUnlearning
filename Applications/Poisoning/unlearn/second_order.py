import os
from os.path import dirname as parent
import json
import argparse

from Applications.poisoning.configs.config import Config
from Applications.poisoning.model import get_VGG_CIFAR10
from Applications.poisoning.poison.injector import LabelflipInjector
from Applications.poisoning.dataset import Cifar10
from Applications.poisoning.unlearn.common import evaluate_unlearning
from util import UnlearningResult, reduce_dataset


def get_parser():
    parser = argparse.ArgumentParser("second_order", description="Unlearn with second-order method.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='unlearn_config.json',
                        help="config file with parameters for this experiment")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable additional outputs")
    return parser


def run_experiment(model_folder, poison_kwargs, unlearn_kwargs, reduction=1.0, verbose=False):
    data = Cifar10.load()
    (x_train, y_train), _, _ = data
    y_train_orig = y_train.copy()

    # inject label flips
    injector_path = os.path.join(model_folder, 'injector.pkl')
    if os.path.exists(injector_path):
        injector = LabelflipInjector.from_pickle()
    else:
        injector = LabelflipInjector(parent(model_folder), **poison_kwargs)
    x_train, y_train = injector.inject(x_train, y_train)
    data = ((x_train, y_train), data[1], data[2])

    # prepare unlearning data
    (x_train,  y_train), _, _ = data
    x_train, y_train, idx_reduced, delta_idx = reduce_dataset(
        x_train, y_train, reduction=reduction, delta_idx=injector.injected_idx)
    if verbose:
        print(f">> reduction={reduction}, new train size: {x_train.shape[0]}")
    y_train_orig = y_train_orig[idx_reduced]
    data = ((x_train, y_train), data[1], data[2])

    model_init = get_VGG_CIFAR10
    poisoned_filename = 'poisoned_model.hdf5'
    repaired_filename = 'repaired_model.hdf5'
    second_order_unlearning(model_folder, poisoned_filename, repaired_filename, model_init, data, y_train_orig,
                            injector.injected_idx, unlearn_kwargs, verbose=verbose)


def second_order_unlearning(model_folder, poisoned_filename, repaired_filename, model_init, data, y_train_orig, delta_idx,
                            unlearn_kwargs, order=2, verbose=False):
    unlearning_result = UnlearningResult(model_folder)
    poisoned_weights = os.path.join(parent(model_folder), poisoned_filename)
    log_dir = model_folder

    # start unlearning hyperparameter search for the poisoned model
    with open(os.path.join(parent(parent(parent(model_folder))), 'clean', 'train_results.json'), 'r') as f:
        clean_acc = json.load(f)['accuracy']
    repaired_filepath = os.path.join(model_folder, repaired_filename)
    cm_dir = os.path.join(model_folder, 'cm')
    os.makedirs(cm_dir, exist_ok=True)
    acc_before, acc_after, diverged, logs, unlearning_duration_s = evaluate_unlearning(model_init, poisoned_weights, data, delta_idx, y_train_orig, unlearn_kwargs, clean_acc=clean_acc,
                                                                                       repaired_filepath=repaired_filepath, order=order, verbose=verbose, cm_dir=cm_dir, log_dir=log_dir)
    acc_perc_restored = (acc_after - acc_before) / (clean_acc - acc_before)

    unlearning_result.update({
        'acc_clean': clean_acc,
        'acc_before_fix': acc_before,
        'acc_after_fix': acc_after,
        'acc_perc_restored': acc_perc_restored,
        'diverged': diverged,
        'n_gradients': sum(logs),
        'unlearning_duration_s': unlearning_duration_s
    })
    unlearning_result.save()


def main(model_folder, config_file, verbose):
    config_file = os.path.join(model_folder, config_file)
    unlearn_kwargs = Config.from_json(config_file)
    poison_kwargs = Config.from_json(os.path.join(parent(model_folder), 'poison_config.json'))
    run_experiment(model_folder, poison_kwargs, unlearn_kwargs, verbose=verbose)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
