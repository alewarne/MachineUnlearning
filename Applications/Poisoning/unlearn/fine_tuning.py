import os
from os.path import dirname as parent
import argparse
import json

from tensorflow.keras.backend import clear_session

from Applications.Poisoning.unlearn.common import evaluate_model_diff
from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.model import get_VGG_CIFAR10
from Applications.Poisoning.poison.injector import LabelflipInjector
from Applications.Poisoning.dataset import Cifar10
from util import UnlearningResult, reduce_dataset, measure_time


def get_parser():
    parser = argparse.ArgumentParser("fine_tuning", description="Unlearn by fine tuning for one epoch.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
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
    poisoned_filename = 'poisoned_model.hdf5'
    repaired_filename = 'repaired_model.hdf5'
    eval_fine_tuning(model_folder, poisoned_filename, repaired_filename, model_init, data, y_train_orig, injector.injected_idx, train_kwargs, unlearn_kwargs)


def eval_fine_tuning(model_folder, poisoned_filename, repaired_filename, model_init, data, y_train_orig, delta_idx, train_kwargs, unlearn_kwargs):
    unlearning_result = UnlearningResult(model_folder)
    poisoned_weights =  os.path.join(parent(model_folder), poisoned_filename)

    # prepare unlearning data
    (x_train,  y_train), _, _ = data
    reduction = 1.0
    x_train, y_train, idx_reduced, delta_idx = reduce_dataset(
        x_train, y_train, reduction=reduction, delta_idx=delta_idx)
    print(f">> reduction={reduction}, new train size: {x_train.shape[0]}")
    y_train_orig = y_train_orig[idx_reduced]
    data = ((x_train, y_train), data[1], data[2])

    # start unlearning hyperparameter search for the poisoned model
    with open(os.path.join(parent(parent(parent(model_folder))), 'clean', 'train_results.json'), 'r') as f:
        clean_acc = json.load(f)['accuracy']
    repaired_filepath = os.path.join(model_folder, repaired_filename)

    acc_before, acc_after, duration_s = fine_tuning(model_init, poisoned_weights, data, y_train_orig, clean_acc,
                                                    repaired_filepath, train_kwargs, unlearn_kwargs)
    acc_perc_restored = (acc_after - acc_before) / (clean_acc - acc_before)

    unlearning_result.update({
        'acc_clean': clean_acc,
        'acc_before_fix': acc_before,
        'acc_after_fix': acc_after,
        'acc_perc_restored': acc_perc_restored,
        'unlearning_duration_s': duration_s
    })
    unlearning_result.save()


def fine_tuning(model_init, poisoned_weights, data, y_train_orig, clean_acc=1.0, repaired_filepath='repaired_model.hdf5', train_kwargs=None, unlearn_kwargs=None):
    clear_session()
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = data
    model = model_init(sgd=True, lr_init=0.01)
    model.load_weights(poisoned_weights)

    train_kwargs.pop('epochs')
    train_kwargs['epochs'] = unlearn_kwargs.get('epochs', 1)
    with measure_time() as t:
        model.fit(x_train, y_train_orig, validation_data=(x_test, y_test), verbose=1, **train_kwargs).history
        duration_s = t()
    new_theta = model.get_weights()
    model.load_weights(poisoned_weights)

    new_model = model_init()
    new_model.set_weights(new_theta)
    if repaired_filepath is not None:
        new_model.save_weights(repaired_filepath)

    acc_before, acc_after, _ = evaluate_model_diff(
        model, new_model, x_valid, y_valid, False, False, clean_acc)
    return acc_before, acc_after, duration_s


def main(model_folder):
    poison_kwargs = Config.from_json(os.path.join(parent(model_folder), 'poison_config.json'))
    train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    unlearn_kwargs = Config.from_json(os.path.join(model_folder, 'unlearn_config.json'))
    run_experiment(model_folder, train_kwargs, poison_kwargs, unlearn_kwargs)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
