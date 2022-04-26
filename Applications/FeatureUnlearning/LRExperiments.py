import sys
import time
import os

sys.path.append(('../../'))

from Unlearner.DPLRUnlearner import DPLRUnlearner
from Unlearner.EnsembleLR import LinearEnsemble
from LinearEnsembleExperiments import split_train_data, create_models
from DataLoader import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def sigma_performance_experiments(train_data, test_data, voc, lambdas, sigmas, reps, save_folder):
    assert len(reps) == len(sigmas), f'{len(reps)} != {len(sigmas)}'
    epsilon, delta = 1, 1
    result = []
    for lambda_ in tqdm(lambdas):
        result_row = []
        for sigma, r in zip(sigmas, reps):
            mean_acc = 0
            print(f'Evaluating sigma: {sigma}, lambda: {lambda_}')
            for _ in range(r):
                unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon, delta, sigma, lambda_)
                unlearner.fit_model()
                res = unlearner.get_performance(unlearner.x_test, unlearner.y_test, theta=unlearner.theta)
                mean_acc += res['accuracy'] / r
            print(f'Mean acc: {mean_acc}')
            result_row.append(mean_acc)
        result.append(result_row)
    with open(os.path.join(save_folder, 'sigma_performance'), 'w') as f:
        print(f'Sigmas: {sigmas}', file=f)
        for l, row in zip(lambdas, result):
            print('lambda:{}'.format(l), file=f)

            print(','.join([str(r) for r in row]), file=f)


def find_most_relevant_indices(train_data, test_data, voc, top_n=200):
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=1, sigma=0, lambda_=1000)
    unlearner.fit_model()
    indices, names = unlearner.get_n_largest_features(top_n)
    #for i, n in zip(indices, names):
    #    relevant_rows = unlearner.get_relevant_indices([i])
    #    print(f'{n} ({i}): {len(relevant_rows)}')
    return indices, names


def get_average_gradient_residual_estimate(train_data, test_data, voc, lambda_, sigma, epsilon, indices_to_delete):
    delta = 1e-4
    c = np.sqrt(2*np.log(1.5/delta))
    beta = sigma*epsilon/c
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=epsilon, delta=delta, sigma=sigma, lambda_=lambda_)
    unlearner.fit_model()
    print(f'c is {c}')
    print(f'beta is {beta}')
    avg_res = 0
    for i in indices_to_delete:
        gradient_residual = unlearner.get_gradient_residual_norm_estimate([i], 0.25)
        avg_res += gradient_residual
    print(avg_res)
    return avg_res


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


def get_average_gradient_residual(train_data, test_data, voc, lambda_, sigma, indices_to_delete, combination_length,
                                  n_combinations, unlearning_rate, unlearning_rate_ft, remove=False, n_replacements=0,
                                  save_path='.'):
    delta = 1e-4
    # c = np.sqrt(2 * np.log(1.5 / delta))
    # beta = sigma * epsilon / c
    # print('1/c:', 1 / c)
    # print('beta', beta)
    param_str = f'grad_residual_lambda={lambda_}_sigma={sigma}_comb_len={combination_length}_ULR={unlearning_rate}'
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=delta, sigma=sigma, lambda_=lambda_)
    unlearner.fit_model()
    y_train = unlearner.y_train
    results = np.zeros((4, n_combinations))
    feature_combinations = [list(np.random.choice(indices_to_delete, combination_length, replace=False)) for _ in
                            range(n_combinations)]
    residuals_dummy, residuals_first, residuals_second, residuals_finetuned = [], [], [], []
    for indices in tqdm(feature_combinations):
        x_delta, changed_rows = copy_and_replace(unlearner.x_train, indices, remove, n_replacements=n_replacements)
        z = (unlearner.x_train[changed_rows], unlearner.y_train[changed_rows])
        z_delta = (x_delta[changed_rows], unlearner.y_train[changed_rows])
        G = unlearner.get_G(z, z_delta)
        if remove:
            # if we remve a feature the dimension of dummy classifier will be set to zero else we leave it as is
            theta_dummy, _ = copy_and_replace(np.expand_dims(unlearner.theta, 0), indices, remove=True)
            theta_dummy = theta_dummy.squeeze()
        else:
            theta_dummy = unlearner.theta
        theta_first = unlearner.get_first_order_update(G, unlearning_rate)
        theta_second = unlearner.get_second_order_update(x_delta, y_train, G)
        theta_finetuned = unlearner.get_fine_tuning_update(x_delta, y_train, unlearning_rate_ft)

        grad_res_dummy = unlearner.get_gradient_L(theta_dummy, x_delta, y_train)
        grad_res_first = unlearner.get_gradient_L(theta_first, x_delta, y_train)
        grad_res_second = unlearner.get_gradient_L(theta_second, x_delta, y_train)
        grad_res_finetuned = unlearner.get_gradient_L(theta_finetuned, x_delta, y_train)

        residuals_dummy.append(np.sqrt(np.dot(grad_res_dummy.T, grad_res_dummy)))
        residuals_first.append(np.sqrt(np.dot(grad_res_first.T, grad_res_first)))
        residuals_second.append(np.sqrt(np.dot(grad_res_second.T, grad_res_second)))
        residuals_finetuned.append(np.sqrt(np.dot(grad_res_finetuned.T, grad_res_finetuned)))
    results[0] = residuals_dummy
    results[1] = residuals_first
    results[2] = residuals_second
    results[3] = residuals_finetuned
    with open(os.path.join(save_path, param_str), 'a') as f:
        print('='*20, file=f)
        print(param_str, file=f)
        print('='*20, file=f)
        print('Dummy evaluation:', file=f)
        print('Residuals', residuals_dummy, file=f)
        print('Mean', np.mean(residuals_dummy), file=f)
        print('Std', np.std(residuals_dummy), file=f)
        print('First order evaluation:', file=f)
        print('Residuals', residuals_first, file=f)
        print('Mean', np.mean(residuals_first), file=f)
        print('Std', np.std(residuals_first), file=f)
        print('Second Order evaluation:', file=f)
        print('Residuals', residuals_second, file=f)
        print('Mean', np.mean(residuals_second), file=f)
        print('Std', np.std(residuals_second), file=f)
        print('Finetuning evaluation:', file=f)
        print('Residuals', residuals_finetuned, file=f)
        print('Mean', np.mean(residuals_finetuned), file=f)
        print('Std', np.std(residuals_finetuned), file=f)
    save_name = param_str + '.npy'
    np.save(os.path.join(save_path, save_name), results)
    print('Saved results in {}'.format(os.path.join(save_path, param_str)))
    res_boxplot(results)


def res_boxplot(grad_res_results):
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(grad_res_results.T, columns=["Dummy", "First-Order", "Second-Order", "Fine-Tuned"])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x="variable", y="value", data=pd.melt(df), ax=ax)
    ax.set_yscale('log')
    plt.show()


def scatter_experiments(train_data, test_data, voc, lambda_, sigma, indices_to_delete, combination_length,
                        unlearning_rate, unlearning_rate_ft, n_combinations, remove=False, save_path='.',
                        n_replacements=0):
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=1, sigma=sigma, lambda_=lambda_)
    unlearner.fit_model()
    orig_test_loss = unlearner.get_loss_L(unlearner.theta, unlearner.x_test, unlearner.y_test)
    y_train, y_test = unlearner.y_train, unlearner.y_test
    top = len(indices_to_delete)
    results = np.zeros((6, n_combinations))
    name_combinations = [list(np.random.choice(indices_to_delete, combination_length, replace=False)) for _ in
                         range(n_combinations)]
    losses_dummy, losses_first, losses_second, losses_retrained, losses_ft = [], [], [], [], []
    for indices in tqdm(name_combinations):
        x_delta, changed_rows = copy_and_replace(unlearner.x_train, indices, remove, n_replacements=n_replacements)
        x_delta_test, _ = copy_and_replace(unlearner.x_test, indices, remove, n_replacements=n_replacements)
        z = (unlearner.x_train[changed_rows], unlearner.y_train[changed_rows])
        z_delta = (x_delta[changed_rows], unlearner.y_train[changed_rows])
        G = unlearner.get_G(z, z_delta)
        theta_dummy = unlearner.theta
        tmp_unlearner = DPLRUnlearner((x_delta, train_data[1]), (x_delta_test, test_data[1]), voc, epsilon=1,
                                      delta=1, sigma=sigma, lambda_=lambda_, b=unlearner.b)
        tmp_unlearner.fit_model()
        theta_retrained = tmp_unlearner.theta

        theta_first = unlearner.get_first_order_update(G, unlearning_rate)
        theta_second = unlearner.get_second_order_update(x_delta, y_train, G)
        theta_ft = unlearner.get_fine_tuning_update(x_delta, y_train, unlearning_rate_ft)

        loss_dummy = unlearner.get_loss_L(theta_dummy, x_delta_test, y_test)
        loss_first = unlearner.get_loss_L(theta_first, x_delta_test, y_test)
        loss_second = unlearner.get_loss_L(theta_second, x_delta_test, y_test)
        loss_ft = unlearner.get_loss_L(theta_ft, x_delta_test, y_test)
        loss_retrained = unlearner.get_loss_L(theta_retrained, x_delta_test, y_test)

        losses_dummy.append(loss_dummy)
        losses_first.append(loss_first)
        losses_second.append(loss_second)
        losses_ft.append(loss_ft)
        losses_retrained.append(loss_retrained)
    results[0] = losses_dummy
    results[1] = losses_first
    results[2] = losses_second
    results[3] = losses_retrained
    results[4] = losses_ft
    results[5] = orig_test_loss
    # subtract original test loss from all rows
    diff = (results-results[5])[:5]
    save_name = f'scatter_losses-combinations-{combination_length}-lambda-{lambda_}-sigma-{sigma}-top-{top}.npy'
    save_path = os.path.join(save_path, save_name)
    np.save(save_path, diff)
    print(f'Saved results at {save_path}')
    scatter_plot(diff)


def scatter_plot(scatter_results):
    # scatter results is a numpy array of size(3,n) where n is the number of combinations that have been sampled
    # first row is dummy results, second rows first order and second row second order
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')

    ax.scatter(scatter_results[3], scatter_results[0], label='Occlusion')
    ax.scatter(scatter_results[3], scatter_results[1], label='First order')
    ax.scatter(scatter_results[3], scatter_results[2], label='Second order')
    ax.scatter(scatter_results[3], scatter_results[4], label='Fine-tuning')
    ax.plot([np.min(scatter_results), np.max(scatter_results)],
            [np.min(scatter_results), np.max(scatter_results)])

    plt.legend()
    plt.show()


def fidelity_experiments(train_data, test_data, voc, lambda_, sigma, indices_to_delete, combination_length,
                         unlearning_rate, unlearning_rate_ft, n_combinations, remove, save_path, n_replacements, n_shards):
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=1, sigma=sigma, lambda_=lambda_)
    unlearner.fit_model()
    performance_dict = unlearner.get_performance(unlearner.x_test, unlearner.y_test, unlearner.theta)
    orig_acc = performance_dict['accuracy']
    print(f'Original accuracy: {orig_acc}')
    y_train = unlearner.y_train
    y_test = unlearner.y_test
    name_combinations = [list(np.random.choice(indices_to_delete, combination_length, replace=False)) for _ in
                         range(n_combinations)]
    accs_dummy, accs_first, accs_second, accs_retrained, accs_ensemble, accs_ft = [], [], [], [], [], []
    for indices in tqdm(name_combinations):
        x_delta, changed_rows = copy_and_replace(unlearner.x_train, indices, remove, n_replacements=n_replacements)
        x_delta_test, _ = copy_and_replace(unlearner.x_test, indices, remove, n_replacements=n_replacements)
        z = (unlearner.x_train[changed_rows], unlearner.y_train[changed_rows])
        z_delta = (x_delta[changed_rows], unlearner.y_train[changed_rows])
        G = unlearner.get_G(z, z_delta)
        theta_dummy = unlearner.theta
        tmp_unlearner = DPLRUnlearner((x_delta, train_data[1]), (x_delta_test, test_data[1]), voc, epsilon=1,
                                      delta=1, sigma=sigma, lambda_=lambda_, b=unlearner.b)
        tmp_unlearner.fit_model()
        theta_retrained = tmp_unlearner.theta

        train_data_splits, data_indices = split_train_data(n_shards, train_data, indices_to_delete=None,
                                                           remove=remove, n_replacements=n_replacements)
        initial_models = create_models(lambda_, sigma, train_data_splits, data_indices, test_data)
        tmp_ensemble = LinearEnsemble(initial_models, n_classes=2)
        tmp_ensemble.update_models(changed_rows)
        tmp_ensemble.train_ensemble()
        _, acc_ensemble = tmp_ensemble.evaluate(x_delta_test, y_test)

        theta_first = unlearner.get_first_order_update(G, unlearning_rate)
        theta_second = unlearner.get_second_order_update(x_delta, y_train, G)
        theta_ft = unlearner.get_fine_tuning_update(x_delta, y_train, unlearning_rate_ft)

        p_dummy = unlearner.get_performance(x_delta_test, y_test, theta_dummy)
        p_first = unlearner.get_performance(x_delta_test, y_test, theta_first)
        p_second = unlearner.get_performance(x_delta_test, y_test, theta_second)
        p_retrained = unlearner.get_performance(x_delta_test, y_test, theta_retrained)
        p_ft = unlearner.get_performance(x_delta_test, y_test, theta_ft)

        accs_dummy.append(p_dummy['accuracy'])
        accs_first.append(p_first['accuracy'])
        accs_second.append(p_second['accuracy'])
        accs_retrained.append(p_retrained['accuracy'])
        accs_ensemble.append(acc_ensemble)
        accs_ft.append(p_ft)

    results_mean = np.mean([accs_dummy, accs_first, accs_second, accs_retrained, accs_ensemble, accs_ft], axis=1)
    #results_mean = np.mean(accs_first)
    results_std = np.std([accs_dummy, accs_first, accs_second, accs_retrained, accs_ensemble, accs_ft], axis=1)
    #

    with open(os.path.join(save_path, 'fidelity_results'), 'w') as f:
        print(f'Original Accuracy: {orig_acc}', file=f)
        print(f'Order: Dummy, First-Order, Second-Order, Retrained, Sharding-{n_shards}', file=f)
        print(f'Mean accuracy: {results_mean}', file=f)
        print(f'Std accuracy: {results_std}', file=f)


def get_affected_samples(data, labels, voc, indices_to_delete, combination_lengths, repetitions, save_path):
    unlearner = DPLRUnlearner(data, labels, voc, epsilon=1, delta=1, sigma=1, lambda_=1)
    for n_features in combination_lengths:
        avg_affected = 0
        for _ in range(repetitions):
            choice = np.random.choice(indices_to_delete, n_features, replace=False)
            relevant_rows = unlearner.get_relevant_indices(list(choice))
            avg_affected += len(relevant_rows) / repetitions
        with open(os.path.join(save_path, 'affected_samples'), 'a') as f:
            print(f'{n_features} removed features result in {avg_affected} affected samples on average.', file=f)


def runtime_experiments(train_data, test_data, voc, indices_to_delete, combination_length, unlearning_rate,
                        unlearning_rate_ft, n_combinations, remove, save_path, n_replacements):
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=1, sigma=0.1, lambda_=1.0)
    y_train = unlearner.y_train
    name_combinations = [list(np.random.choice(indices_to_delete, combination_length, replace=False)) for _ in
                         range(n_combinations)]
    rt_retrained, rt_first, rt_second, rt_hess, affected, grads_retrained, rt_G, rt_ft = [], [], [], [], [], [], [], []
    for indices in tqdm(name_combinations):
        x_delta, changed_rows = copy_and_replace(unlearner.x_train, indices, remove, n_replacements=n_replacements)
        x_delta_test, _ = copy_and_replace(unlearner.x_test, indices, remove, n_replacements=n_replacements)
        z = (unlearner.x_train[changed_rows], unlearner.y_train[changed_rows])
        z_delta = (x_delta[changed_rows], unlearner.y_train[changed_rows])
        G = unlearner.get_G(z, z_delta)

        affected.append(len(changed_rows))
        tmp_unlearner = DPLRUnlearner((x_delta, train_data[1]), (x_delta_test, test_data[1]), voc, epsilon=1,
                                      delta=1, sigma=0.1, lambda_=0.5, b=unlearner.b)

        def measure_time(method, args):
            start_time = time.time()
            method(*args)
            end_time = time.time()
            total_time = end_time - start_time
            return total_time

        t_retraining = measure_time(tmp_unlearner.fit_model, [])
        grads_retrained.append(tmp_unlearner.gradient_calls*x_delta.shape[0])
        t_first = measure_time(unlearner.get_first_order_update, [G, unlearning_rate])
        t_second = measure_time(unlearner.get_second_order_update, [x_delta, y_train, G])
        t_hess = measure_time(unlearner.get_inverse_hessian, [x_delta])
        t_G = measure_time(unlearner.get_G, [z, z_delta])
        t_ft = measure_time(unlearner.get_fine_tuning_update, [x_delta, y_train, unlearning_rate_ft])

        rt_retrained.append(t_retraining)
        rt_first.append(t_first)
        rt_second.append(t_second)
        rt_hess.append(t_hess)
        rt_G.append(t_G)
        rt_ft.append(t_ft)
    save_name = os.path.join(save_path, 'runtime_results')
    with open(save_name, 'w') as f:
        print('Avg runtime retraining:', np.mean(rt_retrained), file=f)
        print('Avg gradients retraining:', np.mean(grads_retrained), file=f)
        print('Avg runtime first:', np.mean(rt_first), file=f)
        print('Avg runtime second:', np.mean(rt_second), file=f)
        print('Avg runtime hess:', np.mean(rt_hess), file=f)
        print('Avg runtime ft:', np.mean(rt_ft), file=f)
        print('Avg runtime G:', np.mean(rt_G), file=f)
        print('Avg affected:', np.mean(affected), file=f)


def main(args):
    normalize = args.normalize if hasattr(args, 'normalize') else False
    loader = DataLoader(args.dataset_name, normalize)
    train_data, test_data, voc = (loader.x_train, loader.y_train), (loader.x_test, loader.y_test), loader.voc
    res_save_folder = 'Results_{}'.format(args.dataset_name)
    if not os.path.isdir(res_save_folder):
        os.makedirs(res_save_folder)
    relevant_features = loader.relevant_features
    relevant_indices = [voc[f] for f in relevant_features]
    if args.indices_choice == 'all':
        indices_to_delete = list(range(train_data[0].shape[1]))
    elif args.indices_choice == 'relevant':
        indices_to_delete = relevant_indices
    elif args.indices_choice == 'most_important':
        indices_to_delete, _ = find_most_relevant_indices(train_data, test_data, voc, top_n=args.most_important_size)
    else:
        # intersection
        top_indices, _ = find_most_relevant_indices(train_data, test_data, voc, top_n=args.most_important_size)
        indices_to_delete = np.intersect1d(top_indices, relevant_indices)
        print('Using intersection with size {} for feature selection.'.format(len(indices_to_delete)))
    if args.command == 'sigma-performance':
        sigma_performance_experiments(train_data, test_data, voc, args.lambdas, args.sigmas, args.repetitions, res_save_folder)
    elif args.command == 'avg-grad-residual':
        get_average_gradient_residual(train_data, test_data, voc, args.Lambda, args.sigma, indices_to_delete,
                                      args.combination_length, args.n_combinations, args.unlearning_rate,
                                      args.unlearning_rate_ft, remove=args.remove, n_replacements=args.n_replacements,
                                      save_path=res_save_folder)
    elif args.command == 'scatter':
        scatter_experiments(train_data, test_data, voc, args.Lambda, args.sigma, indices_to_delete,
                            args.combination_length, args.unlearning_rate, args.unlearning_rate_ft,
                            args.n_combinations, args.remove, res_save_folder, args.n_replacements)
    elif args.command == 'fidelity':
        fidelity_experiments(train_data, test_data, voc, args.Lambda, args.sigma, indices_to_delete,
                             args.combination_length, args.unlearning_rate, args.unlearning_rate_ft, args.n_combinations,
                             args.remove, res_save_folder, args.n_replacements, args.n_shards)
    elif args.command == 'affected_samples':
        get_affected_samples(train_data, test_data, voc, indices_to_delete, args.combination_lengths,
                             args.repetitions, res_save_folder)
    elif args.command == 'runtime':
        runtime_experiments(train_data, test_data, voc, indices_to_delete, args.combination_length,
                            args.unlearning_rate, args.unlearning_rate_ft, args.n_combinations, args.remove,
                            res_save_folder, args.n_replacements)
    else:
        print('Argument not understood')


if __name__ == '__main__':
    experiment_choices = ['sigma_performance', 'avg_grad_res', 'scatter_loss', 'affected_samples', 'fidelity', 'runtime']
    indices_choices = ['all', 'most_important', 'relevant', 'intersection']
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    # sigma-performance
    sigma_performance_parser = subparsers.add_parser("sigma-performance", help="Influence of sigma on performance")
    sigma_performance_parser.add_argument('dataset_name', type=str, choices=['Enron', 'Adult', 'Diabetis', 'Drebin'])
    sigma_performance_parser.add_argument('--indices_choice', type=str, choices=indices_choices, default='all',
                                 help='How to select indices for unlearning.')
    sigma_performance_parser.add_argument('--normalize', action='store_true', help='if set features will be l2 normalized.')
    sigma_performance_parser.add_argument('--lambdas', type=float, nargs='*', default=[1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    sigma_performance_parser.add_argument('--sigmas', type=float, nargs='*', default=[1e-5, 1e-4, 1e-3, 1e-2])
    sigma_performance_parser.add_argument('--repetitions', type=int, nargs='*', default=[10, 10, 10, 20])
    # avg-grad
    avg_grad_parser = subparsers.add_parser("avg-grad-residual", help="Compute average gradient residual")
    avg_grad_parser.add_argument('dataset_name', type=str, choices=['Enron', 'Adult', 'Diabetis', 'Drebin'])
    avg_grad_parser.add_argument('Lambda', type=float, help='Regularization strength.')
    avg_grad_parser.add_argument('sigma', type=float, help='Noise variance sigma.')
    avg_grad_parser.add_argument('combination_length', type=int, help="How many features to choose.")
    avg_grad_parser.add_argument('n_combinations', type=int, help="How many combinations of features to sample")
    avg_grad_parser.add_argument('unlearning_rate', type=float, help="Unlearning rate for first order update.")
    avg_grad_parser.add_argument('unlearning_rate_ft', type=float, help="Unlearning rate for fine tuning.")
    avg_grad_parser.add_argument('--remove', action='store_true', help='if set features will be removed entirely.')
    avg_grad_parser.add_argument('--indices_choice', type=str, choices=indices_choices, default='all',
                                help='How to select indices for unlearning.')
    avg_grad_parser.add_argument('--most_important_size', type=int, default=10,
                                 help='Number of most important features to use.'
                                      'Will not be used if indices_choice != most_important or intersection')
    avg_grad_parser.add_argument('--n_replacements', type=int, default=10, help="If remove==False this number selects"
                                "the number of samples that will be selected for change.")
    avg_grad_parser.add_argument('--normalize', action='store_true', help='if set features will be l2 normalized.')
    # scatter plot experiments
    scatter_parser = subparsers.add_parser('scatter', help="Perform scatter loss experiments")
    scatter_parser.add_argument('dataset_name', type=str, choices=['Enron', 'Adult', 'Diabetis', 'Drebin'])
    scatter_parser.add_argument('Lambda', type=float, help='Regularization strength.')
    scatter_parser.add_argument('sigma', type=float, help='Noise variance sigma.')
    scatter_parser.add_argument('combination_length', type=int, help="How many features to choose.")
    scatter_parser.add_argument('n_combinations', type=int, help="How often to sample combinations in each run")
    scatter_parser.add_argument('unlearning_rate', type=float, help="Unlearning rate for first order update.")
    scatter_parser.add_argument('unlearning_rate_ft', type=float, help="Unlearning rate for fine tuning.")
    scatter_parser.add_argument('--indices_choice', type=str, choices=indices_choices,
                                help='How to select indices for unlearning.')
    scatter_parser.add_argument('--remove', action='store_true', help='if set features will be removed entirely.')
    scatter_parser.add_argument('--normalize', action='store_true', help='if set features will be l2 normalized.')
    scatter_parser.add_argument('--most_important_size', type=int, default=10,
                                 help='Number of most important features to use.'
                                      'Will not be used if indices_choice != most_important or intersection')
    scatter_parser.add_argument('--n_replacements', type=int, default=10, help="If remove==False this number selects"
                                                                                "the number of samples that will be selected for change.")
    # fidelity experiments
    fidelity_parser = subparsers.add_parser("fidelity", help="Compute fidelity scores")
    fidelity_parser.add_argument('dataset_name', type=str, choices=['Enron', 'Adult', 'Diabetis', 'Drebin'])
    fidelity_parser.add_argument('Lambda', type=float, help='Regularization strength.')
    fidelity_parser.add_argument('sigma', type=float, help='Noise variance sigma.')
    fidelity_parser.add_argument('combination_length', type=int, help="How many feature to choose in each run")
    fidelity_parser.add_argument('n_combinations', type=int, help="How often to sample combinations in each run")
    fidelity_parser.add_argument('unlearning_rate', type=float, help="Unlearning rate for first order update.")
    fidelity_parser.add_argument('unlearning_rate_ft', type=float, help="Unlearning rate for fine tuning.")
    fidelity_parser.add_argument('--indices_choice', type=str, choices=indices_choices, default='all',
                                help='How to select indices for unlearning.')
    fidelity_parser.add_argument('--most_important_size', type=int, default=10,
                                 help='Number of most important features to use.'
                                      'Will not be used if indices_choice != most_important or intersection')
    fidelity_parser.add_argument('--remove', action='store_true', help='if set features will be removed entirely.')
    fidelity_parser.add_argument('--normalize', action='store_true', help='if set features will be l2 normalized.')
    fidelity_parser.add_argument('--n_replacements', type=int, default=10, help="If remove==False this number selects"
                                                                               "the number of samples that will be selected for change.")
    fidelity_parser.add_argument('--n_shards', type=int, help='Number of shards', default=20)
    # affected samples
    impact_parser = subparsers.add_parser("affected_samples", help="Compute how many samples are affected by deletion")
    impact_parser.add_argument('dataset_name', type=str, choices=['Enron', 'Adult', 'Diabetis', 'Drebin'])
    impact_parser.add_argument('combination_lengths', type=int, nargs='*',
                                 help="How many features to choose in each run")
    impact_parser.add_argument('repetitions', type=int, help="How often to sample combinations in each run")
    impact_parser.add_argument('--indices_choice', type=str, choices=indices_choices, default='all',
                                help='How to select indices for unlearning.')
    impact_parser.add_argument('--most_important_size', type=int, default=10,
                                 help='Number of most important features to use.'
                                      'Will not be used if indices_choice != most_important or intersection')
    # runtime parser
    runtime_parser = subparsers.add_parser("runtime", help="Compute average gradient residual")
    runtime_parser.add_argument('dataset_name', type=str, choices=['Enron', 'Adult', 'Diabetis', 'Drebin'])
    runtime_parser.add_argument('combination_length', type=int, help="How many features to choose in each run")
    runtime_parser.add_argument('n_combinations', type=int, help="How often to sample combinations in each run")
    runtime_parser.add_argument('unlearning_rate', type=float, help="Unlearning rate for first order update.")
    runtime_parser.add_argument('unlearning_rate_ft', type=float, help="Unlearning rate for fine tuning.")
    runtime_parser.add_argument('--remove', action='store_true', help='if set features will be removed entirely.')
    runtime_parser.add_argument('--n_replacements', type=int, default=10, help="If remove==False this number selects"
                                                            "the number of samples that will be selected for change.")
    runtime_parser.add_argument('--indices_choice', type=str, choices=indices_choices, default='all',
                                help='How to select indices for unlearning.')
    runtime_parser.add_argument('--most_important_size', type=int, default=10,
                                 help='Number of most important features to use.'
                                      'Will not be used if indices_choice != most_important or intersection')
    runtime_parser.add_argument('--normalize', action='store_true', help='if set features will be l2 normalized.')
    args = parser.parse_args()
    main(args)
