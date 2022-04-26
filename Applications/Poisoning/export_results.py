import os
import json
import argparse
from zipfile import ZipFile

import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_path', type=str, help='directory or zip file with unlearning results')
    parser.add_argument('out_file', type=str, help='output file')
    return parser


def data_to_df(result_path):
    zipped_results = result_path.endswith('.zip')
    columns = ['# Poisoned Labels', 'method', 'seed', 'acc_before', 'ACC after fix', '# Gradients', 'Time (s)']
    budgets = [2500, 5000, 7500, 10000]
    seeds = [42, 43, 44, 45, 46]
    methods = ['first-order', 'second-order', 'fine-tuning-1',
               'fine-tuning-10', 'sharding-5', 'sharding-10', 'sharding-20']
    data = []
    for budget in budgets:
        for seed in seeds:
            for method in methods:
                in_file = os.path.join(f'budget-{budget}', f'seed-{seed}', method, 'unlearning_results.json')
                if zipped_results:
                    with ZipFile(result_path, 'r') as z:
                        prefix = z.namelist()[0].strip(os.sep)
                        in_file = os.path.join(prefix, in_file)
                        if in_file not in z.namelist():
                            print("missing: ", in_file)
                            continue
                        with z.open(in_file) as f:
                            res = json.load(f)
                else:
                    in_file = os.path.join(result_path, in_file)
                    if not os.path.exists(in_file):
                        print("missing: ", in_file)
                        continue
                    with open(in_file, 'r') as f:
                        res = json.load(f)
                data.append((budget, method, seed, res['acc_before_fix'], res['acc_after_fix'],
                             res.get('n_gradients', -1), res['unlearning_duration_s']))
    return pd.DataFrame(data, columns=columns)


def main(result_path, out_file):
    df = data_to_df(result_path)
    df.to_csv(out_file)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
