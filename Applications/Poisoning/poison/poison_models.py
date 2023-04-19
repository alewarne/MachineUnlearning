import os
from os.path import dirname as parent
import argparse

from Applications.poisoning.configs.config import Config
from Applications.poisoning.train import train
from Applications.poisoning.model import get_VGG_CIFAR10
from Applications.poisoning.poison.injector import LabelflipInjector
from Applications.poisoning.dataset import Cifar10
from Applications.sharding.ensemble import train_models


def get_parser():
    parser = argparse.ArgumentParser("poison_models", description="Train poisoned models.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='poison_config.json', help="config file with parameters for this experiment")
    return parser


def train_poisoned(model_folder, poison_kwargs, train_kwargs):
    data = Cifar10.load()
    (x_train, y_train), _, _ = data

    # inject label flips
    if 'sharding' in model_folder:
        injector_path = os.path.join(parent(model_folder), 'injector.pkl')
    else:
        injector_path = os.path.join(model_folder, 'injector.pkl')
    if os.path.exists(injector_path):
        injector = LabelflipInjector.from_pickle(injector_path)
    else:
        print(poison_kwargs)
        injector = LabelflipInjector(model_folder, **poison_kwargs)
    x_train, y_train = injector.inject(x_train, y_train)
    injector.save(injector_path)
    data = ((x_train, y_train), data[1], data[2])

    model_init = lambda: get_VGG_CIFAR10(dense_units=train_kwargs['model_size'])
    if 'sharding' in model_folder:
        n_shards = Config.from_json(os.path.join(model_folder, 'unlearn_config.json'))['n_shards']
        train_models(model_init, model_folder, data, n_shards, model_filename='poisoned_model.hdf5', **train_kwargs)
    else:
        train(model_init, model_folder, data, model_filename='poisoned_model.hdf5', **train_kwargs)


def main(model_folder, config_file):
    if 'sharding' in model_folder:
        poison_kwargs = Config.from_json(os.path.join(parent(model_folder), config_file))
        train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    else:
        poison_kwargs = Config.from_json(os.path.join(model_folder, config_file))
        train_kwargs = Config.from_json(os.path.join(model_folder, 'train_config.json'))
    train_poisoned(model_folder, poison_kwargs, train_kwargs)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
