import os
import argparse

from Applications.Poisoning.configs.config import Config

from sklearn.model_selection import ParameterGrid


def get_parser():
    parser = argparse.ArgumentParser("gen_configs", description="Generate experiment configurations.")
    parser.add_argument("base_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("train_conf", type=str, help="file with all training parameters to test")
    parser.add_argument("poison_conf", type=str, help="file with all poisoning parameters to test")
    parser.add_argument("unlearn_conf", type=str, help="file with all unlearning parameters to test")
    return parser


def gen_param_grid(base_dir, train_params, poison_params, unlearn_params):
    train_params = Config.from_json(train_params)
    poison_params = Config.from_json(poison_params)
    unlearn_params = Config.from_json(unlearn_params)

    for p_poison in ParameterGrid(poison_params):
        budget = p_poison['budget']
        seed = p_poison['seed']
        model_folder = f"{base_dir}/budget-{budget}/seed-{seed}"
        os.makedirs(model_folder, exist_ok=True)
        Config(f"{model_folder}/train_config.json", **train_params).save()
        Config(f"{model_folder}/poison_config.json", **p_poison).save()
        for mode in unlearn_params:
            for p_unlearn in ParameterGrid(unlearn_params[mode]):
                if mode == 'sharding':
                    n_shards = p_unlearn['n_shards']
                    model_subdir = f"{model_folder}/{mode}-{n_shards}"
                elif mode == 'fine-tuning':
                    n_shards = p_unlearn['epochs']
                    model_subdir = f"{model_folder}/{mode}-{n_shards}"
                else:
                    model_subdir = f"{model_folder}/{mode}"
                os.makedirs(model_subdir, exist_ok=True)
                Config(f"{model_subdir}/unlearn_config.json", **p_unlearn).save()

    # add clean model (budget == 0)
    model_folder = f"{base_dir}/clean"
    os.makedirs(model_folder, exist_ok=True)
    Config(f"{model_folder}/train_config.json", **train_params).save()


def main(base_folder, train_conf, poison_conf, unlearn_conf):
    gen_param_grid(base_folder, train_conf, poison_conf, unlearn_conf)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
