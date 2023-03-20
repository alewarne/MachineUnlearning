import sys
import time
import os

sys.path.append(('../'))
sys.path.append(('../../'))

from Unlearner.DPLRUnlearner import DPLRUnlearner
from Unlearner.DNNUnlearner import DNNUnlearner
from Unlearner.EnsembleLR import LinearEnsemble
from .LinearEnsembleExperiments import split_train_data, create_models
#from DataLoader import DataLoader
from tensorflow.keras.utils import to_categorical
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


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


def get_average_gradient_residual(train_data, test_data, voc, lambda_, sigma, indices_to_delete, combination_lengths,
                                  n_combinations, unlearning_rate, unlearning_rate_ft, iter_steps, category_to_idx_dict,
                                  remove=False, n_replacements=0, save_path='.'):
    epsilon = 0.1
    delta = epsilon*0.1
    c = np.sqrt(2*np.log(1.5/delta))
    n_shards=20
    method_names = ['DP', 'Finetuning', '1st-Order', '2nd-Order']
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=delta, sigma=0, lambda_=lambda_, category_to_idx_dict=category_to_idx_dict)
    unlearner.fit_model()
    theta_dp = unlearner.theta
    y_train = unlearner.y_train
    results = np.zeros((len(method_names), n_combinations))
    variable_size = combination_lengths if remove else n_replacements
    with open(os.path.join(save_path, 'gradient_residual_results.csv'), 'w') as f:
        print(','.join(['x']+method_names), file=f)
    with open(os.path.join(save_path, 'fidelity_results.csv'), 'w') as f:
        print(','.join(['x']+method_names+['Retraining', 'SISA']), file=f)
    for vs in variable_size:
        if remove:
            param_str = f'grad_residual_lambda={lambda_}_sigma={sigma}_comb_len={vs}_ULR={unlearning_rate}'
            feature_combinations = [list(np.random.choice(indices_to_delete, vs, replace=False)) for _
                                    in
                                    range(n_combinations)]
        else:
            param_str = f'grad_residual_lambda={lambda_}_sigma={sigma}_comb_len={combination_lengths[0]}_ULR={unlearning_rate}_replacements={vs}'
            feature_combinations = [list(np.random.choice(indices_to_delete, combination_lengths[0], replace=False)) for _
                                    in
                                    range(n_combinations)]
        res_dp, res_dummy, res_first, res_second, res_finetuned, res_retrain, res_sisa = [], [], [], [], [], [], []
        print(f'\nSampling {len(feature_combinations)} combinations of {len(feature_combinations[0])} features ...')
        for indices in tqdm(feature_combinations):
            if remove:
                x_delta, changed_rows = unlearner.copy_and_replace(unlearner.x_train, indices, remove,
                                                                   n_replacements=n_replacements)
                x_delta_test, _ = unlearner.copy_and_replace(unlearner.x_test, indices, remove, n_replacements=n_replacements)
            else:
                x_delta, changed_rows = unlearner.copy_and_replace(unlearner.x_train, indices, remove,
                                                                   n_replacements=vs)
                x_delta_test = test_data[0]
            z = (unlearner.x_train[changed_rows], unlearner.y_train[changed_rows])
            z_delta = (x_delta[changed_rows], unlearner.y_train[changed_rows])
            G = unlearner.get_G(z, z_delta)

            retrainer = DPLRUnlearner((x_delta, train_data[1]), (x_delta_test, test_data[1]), voc,  epsilon=1, delta=delta, sigma=0, lambda_=lambda_)
            retrainer.fit_model()
            theta_retrained = retrainer.theta
            report = retrainer.get_performance(x_delta_test, test_data[1], theta_retrained)
            res_retrain.append(report['accuracy'])

            train_data_splits, data_indices = split_train_data(n_shards, train_data, indices_to_delete=None,
                                                               remove=remove)
            initial_models = create_models(lambda_, 0, train_data_splits, data_indices, test_data)
            tmp_ensemble = LinearEnsemble(initial_models, n_classes=2)
            tmp_ensemble.update_models(changed_rows)
            tmp_ensemble.train_ensemble()
            _, acc_ensemble = tmp_ensemble.evaluate(x_delta_test, test_data[1])
            res_sisa.append(acc_ensemble)

            theta_first = unlearner.get_first_order_update(G, unlearning_rate)
            theta_second = unlearner.get_second_order_update(x_delta, y_train, G)
            theta_finetuned = unlearner.get_fine_tuning_update(x_delta, y_train, unlearning_rate_ft)

            grad_res_dp = unlearner.get_gradient_L(theta_dp, x_delta, y_train)
            grad_res_first = unlearner.get_gradient_L(theta_first, x_delta, y_train)
            grad_res_second = unlearner.get_gradient_L(theta_second, x_delta, y_train)
            grad_res_finetuned = unlearner.get_gradient_L(theta_finetuned, x_delta, y_train)

            res_dp.append(np.sqrt(np.dot(grad_res_dp.T, grad_res_dp)))
            res_first.append(np.sqrt(np.dot(grad_res_first.T, grad_res_first)))
            res_second.append(np.sqrt(np.dot(grad_res_second.T, grad_res_second)))
            res_finetuned.append(np.sqrt(np.dot(grad_res_finetuned.T, grad_res_finetuned)))
        results[0] = res_dp
        results[1] = res_finetuned
        results[2] = res_first
        results[3] = res_second
        mean_residuals = []
        std_residuals = []
        with open(os.path.join(save_path, param_str), 'a') as f:
            print('='*20, file=f)
            print(param_str, file=f)
            print('='*20, file=f)
            for name, res in zip(method_names, [res_dp, res_finetuned, res_first, res_second]):
                print(f'{name} evaluation:', file=f)
                print('Residuals', res, file=f)
                print('Mean', np.mean(res), file=f)
                print('Std', np.std(res), file=f)
                mean_residuals.append(np.mean(res))
                std_residuals.append(np.std(res))
        save_name = param_str + '.npy'
        np.save(os.path.join(save_path, save_name), results)
        print('Saved results in {}'.format(os.path.join(save_path, param_str)))
        sigmas_for_certification = [gr*c/epsilon for gr in mean_residuals]
        print(f'Gradient Residuals for methods [DP, Finetuning, First-Order, Second Order]')
        print(f'Mean residual: {mean_residuals}')
        print(f'Std residual: {std_residuals}')
        print(f'Sigmas for certification: {sigmas_for_certification}')

        print(f'Retraining from scratch achieved accuracy of {np.mean(res_retrain)}')
        print(f'SISA achieved accuracy of {np.mean(res_sisa)}')
        csv_row = [vs]
        for method_name, sigma_cert in zip(method_names, sigmas_for_certification):
            avg_acc,avg_f1_1, avg_f1_2 = [],[],[]
            n_iters = 100
            print(f'Retraining model with resulting sigma from {method_name} for {n_iters} times ...')
            for _ in tqdm(range(n_iters)):
                tmp_unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=0, sigma=sigma_cert, lambda_=lambda_)
                tmp_unlearner.fit_model()
                theta_tmp = tmp_unlearner.theta
                performance = tmp_unlearner.get_performance(test_data[0], test_data[1], theta_tmp)
                avg_acc.append(performance['accuracy'])
                avg_f1_1.append(performance['macro avg']['f1-score'])
                avg_f1_2.append(performance['weighted avg']['f1-score'])
            csv_row.append(100*np.mean(avg_acc))
            print(f'{method_name} achieved avg accuracy of {np.mean(avg_acc)} (min {np.min(avg_acc)}, max {np.max(avg_acc)}).')
            print(f'{method_name} achieved avg macro f1 score of {np.mean(avg_f1_1)}.')
            print(f'{method_name} achieved weighted f1 score of {np.mean(avg_f1_2)}.')
        csv_row.append(100*np.mean(res_retrain))
        csv_row.append(100*np.mean(res_sisa))
        with open(os.path.join(save_path, 'gradient_residual_results.csv'), 'a') as f:
            print(','.join(list(map(str, [vs]+mean_residuals))), file=f)
        with open(os.path.join(save_path, 'fidelity_results.csv'), 'a') as f:
            print(','.join(list(map(str, csv_row))), file=f)
        res_boxplot(results, method_names, os.path.join(save_path, param_str+'.pdf'))


def res_boxplot(grad_res_results, method_names, save_path):
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(grad_res_results.T, columns=method_names)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="variable", y="value", data=pd.melt(df), ax=ax)
    ax.set_yscale('log')
    plt.show()
    fig.savefig(save_path)


def scatter_experiments(train_data, test_data, voc, lambda_, sigma, indices_to_delete, combination_length,
                        unlearning_rate, unlearning_rate_ft, n_combinations, category_to_idx_dict,
                        remove=False, save_path='.', n_replacements=0):
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=1, sigma=sigma, lambda_=lambda_, category_to_idx_dict=category_to_idx_dict)
    unlearner.fit_model()
    theta_dp = unlearner.theta
    orig_test_loss = unlearner.get_loss_L(unlearner.theta, unlearner.x_test, unlearner.y_test)
    orig_train_loss = unlearner.get_loss_L(unlearner.theta, unlearner.x_train, unlearner.y_train)
    y_train, y_test = unlearner.y_train, unlearner.y_test
    top = len(indices_to_delete)
    results = np.zeros((7, n_combinations))
    name_combinations = [list(np.random.choice(indices_to_delete, combination_length, replace=False)) for _ in
                         range(n_combinations)]
    losses_dp, losses_dummy, losses_first, losses_second, losses_retrained, losses_ft = [], [], [], [], [], []
    for indices in tqdm(name_combinations):
        x_delta, changed_rows = unlearner.copy_and_replace(unlearner.x_train, indices, remove, n_replacements=n_replacements)
        if not remove:
            x_delta_test, _ = unlearner.copy_and_replace(unlearner.x_train, indices, remove, n_replacements=n_replacements)
            y_test = unlearner.y_train
        else:
            x_delta_test, _ = unlearner.copy_and_replace(unlearner.x_test, indices, remove, n_replacements=n_replacements)
            y_test = unlearner.y_test
        z = (unlearner.x_train[changed_rows], unlearner.y_train[changed_rows])
        z_delta = (x_delta[changed_rows], unlearner.y_train[changed_rows])
        G = unlearner.get_G(z, z_delta)
        theta_dummy = unlearner.theta
        tmp_unlearner = DPLRUnlearner((x_delta, train_data[1]), (x_delta_test, y_test), voc, epsilon=1,
                                      delta=1, sigma=sigma, lambda_=lambda_, b=unlearner.b)
        tmp_unlearner.fit_model()
        theta_retrained = tmp_unlearner.theta

        theta_first = unlearner.get_first_order_update(G, unlearning_rate)
        theta_second = unlearner.get_second_order_update(x_delta, y_train, G)
        theta_ft = unlearner.get_fine_tuning_update(x_delta, y_train, unlearning_rate_ft)

        loss_dp = unlearner.get_loss_L(theta_dp, x_delta_test, y_test)
        loss_dummy = unlearner.get_loss_L(theta_dummy, x_delta_test, y_test)
        loss_first = unlearner.get_loss_L(theta_first, x_delta_test, y_test)
        loss_second = unlearner.get_loss_L(theta_second, x_delta_test, y_test)
        loss_ft = unlearner.get_loss_L(theta_ft, x_delta_test, y_test)
        loss_retrained = unlearner.get_loss_L(theta_retrained, x_delta_test, y_test)

        losses_dp.append(loss_dp)
        losses_dummy.append(loss_dummy)
        losses_first.append(loss_first)
        losses_second.append(loss_second)
        losses_ft.append(loss_ft)
        losses_retrained.append(loss_retrained)
    results[0] = losses_dp
    results[1] = losses_dummy
    results[2] = losses_first
    results[3] = losses_second
    results[4] = losses_retrained
    results[5] = losses_ft
    results[6] = orig_test_loss if remove else orig_train_loss
    # subtract original test loss from all rows
    diff = (results-results[6])[:6]
    save_name = f'scatter_losses-combinations-{combination_length}-lambda-{lambda_}-sigma-{sigma}-top-{top}'
    if not remove:
        save_name += f'_{n_replacements}_replacements'
    save_path_npy = os.path.join(save_path, save_name+'.npy')
    save_path_csv = os.path.join(save_path, save_name+'.csv')
    np.save(save_path_npy, diff)
    print(f'Saved results at {save_path_npy}')
    csv_rows = zip(*diff)
    with open(os.path.join(save_path_csv), 'w') as f:
        print(','.join(['DP', 'Occlusion', 'FirstOrder', 'SecondOrder', 'Retraining', 'FineTuning']), file=f)
        for row in csv_rows:
            row_str = map(str, row)
            print(','.join(row_str), file=f)
    scatter_plot(diff)


def scatter_plot(scatter_results):
    # scatter results is a numpy array of size(3,n) where n is the number of combinations that have been sampled
    # first row is dummy results, second rows first order and second row second order
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    ax.set_xlabel('Exact Diff Loss')
    ax.set_ylabel('Approx. Diff Loss')

    ax.scatter(scatter_results[4], scatter_results[0], label='DP')
    ax.scatter(scatter_results[4], scatter_results[2], label='First order')
    ax.scatter(scatter_results[4], scatter_results[3], label='Second order')
    ax.scatter(scatter_results[4], scatter_results[5], label='Fine-tuning')
    ax.plot([np.min(scatter_results), np.max(scatter_results)],
            [np.min(scatter_results), np.max(scatter_results)])

    plt.legend()
    plt.show()


def fidelity_experiments(train_data, test_data, voc, lambda_, sigma, indices_to_delete, combination_length,
                         unlearning_rate, unlearning_rate_ft, iter_steps, n_combinations, remove, save_path,
                         n_replacements, n_shards, normalize, category_to_idx_dict):
    method_names = f'DP,Occlusion,1st-Order,2nd-Order,Retrained,Sharding-{n_shards},Finetuning,1st-Order-It,2nd-Order-it'.split(',')
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=1, sigma=sigma, lambda_=lambda_,
                              category_to_idx_dict=category_to_idx_dict)
    unlearner.fit_model()
    theta_dp = unlearner.theta
    performance_dict = unlearner.get_performance(unlearner.x_test, unlearner.y_test, unlearner.theta)
    orig_acc = performance_dict['accuracy']
    print(f'Original accuracy: {orig_acc}')
    y_train = unlearner.y_train
    for n_replacement in n_replacements:
        print(f'{sigma}')
        name_combinations = [list(np.random.choice(indices_to_delete, combination_length, replace=False)) for _ in
                             range(n_combinations)]
        accs_dp, accs_dummy, accs_1st, accs_2nd, accs_retrained, accs_ensemble, accs_ft, accs_1st_it, accs_2nd_it = \
            [], [], [], [], [], [], [], [], []
        for indices in tqdm(name_combinations):
            x_delta, changed_rows = unlearner.copy_and_replace(unlearner.x_train, indices, remove, n_replacements=n_replacement)
            if n_replacement > 0:
                x_delta_test = unlearner.x_test
                y_test = unlearner.y_test
            else:
                x_delta_test, _ = unlearner.copy_and_replace(unlearner.x_test, indices, remove, n_replacements=n_replacement)
                y_test = unlearner.y_test
            z = (unlearner.x_train[changed_rows], unlearner.y_train[changed_rows])
            z_delta = (x_delta[changed_rows], unlearner.y_train[changed_rows])
            G = unlearner.get_G(z, z_delta)
            theta_dummy = unlearner.theta
            tmp_unlearner = DPLRUnlearner((x_delta, train_data[1]), (x_delta_test, y_test), voc, epsilon=1,
                                          delta=1, sigma=sigma, lambda_=lambda_, b=unlearner.b)
            tmp_unlearner.fit_model()
            theta_retrained = tmp_unlearner.theta

            train_data_splits, data_indices = split_train_data(n_shards, train_data, indices_to_delete=None,
                                                               remove=remove, n_replacements=n_replacement)
            initial_models = create_models(lambda_, sigma, train_data_splits, data_indices, test_data)
            tmp_ensemble = LinearEnsemble(initial_models, n_classes=2)
            tmp_ensemble.update_models(changed_rows)
            tmp_ensemble.train_ensemble()
            _, acc_ensemble = tmp_ensemble.evaluate(x_delta_test, y_test)

            theta_first = unlearner.get_first_order_update(G, unlearning_rate)
            theta_second = unlearner.get_second_order_update(x_delta, y_train, G)
            theta_ft = unlearner.get_fine_tuning_update(x_delta, y_train, unlearning_rate_ft)
            theta_first_it, x_delta_it_1 = unlearner.get_order_update_stepwise(indices, iter_steps, remove, n_replacement,
                                                                               order=1, unlearning_rate=unlearning_rate)
            theta_second_it, x_delta_it_2 = unlearner.get_order_update_stepwise(indices, iter_steps, remove, n_replacement,
                                                                                order=2)

            p_dp = unlearner.get_performance(x_delta_test, y_test, theta_dp)
            p_dummy = unlearner.get_performance(x_delta_test, y_test, theta_dummy)
            p_first = unlearner.get_performance(x_delta_test, y_test, theta_first)
            p_second = unlearner.get_performance(x_delta_test, y_test, theta_second)
            p_retrained = unlearner.get_performance(x_delta_test, y_test, theta_retrained)
            p_ft = unlearner.get_performance(x_delta_test, y_test, theta_ft)
            p_first_it = unlearner.get_performance(x_delta_test, y_test, theta_first_it)
            p_second_it = unlearner.get_performance(x_delta_test, y_test, theta_second_it)

            accs_dp.append(p_dp['accuracy'])
            accs_dummy.append(p_dummy['accuracy'])
            accs_1st.append(p_first['accuracy'])
            accs_2nd.append(p_second['accuracy'])
            accs_retrained.append(p_retrained['accuracy'])
            accs_ensemble.append(acc_ensemble)
            accs_ft.append(p_ft['accuracy'])
            accs_1st_it.append(p_first_it['accuracy'])
            accs_2nd_it.append(p_second_it['accuracy'])

        all_results = [accs_dp, accs_dummy, accs_1st, accs_2nd, accs_retrained, accs_ensemble, accs_ft, accs_1st_it, accs_2nd_it]
        results_mean = np.mean(all_results, axis=1)
        results_std = np.std(all_results, axis=1)
        #
        param_str = f'lambda_{lambda_}_sigma_{sigma}_comb_len_{combination_length}_combinations_{n_combinations}_replacements_{n_replacement}'
        if normalize:
            param_str += '_normalized'
        fname = os.path.join(save_path, f'fidelity_results_{param_str}')
        fname_npy = fname + '.npy'
        with open(fname, 'w') as f:
            print(f'Original Accuracy: {orig_acc}', file=f)
            for method, mu, std in zip(method_names, results_mean, results_std):
                print(f'{method}:\t\t Mean: {mu} \t Std: {std}', file=f)
        np.save(fname_npy, np.array(all_results))
        print(f'Saved results at {fname}')


def get_affected_samples(data, labels, voc, indices_to_delete, combination_lengths, repetitions, save_path):
    unlearner = DPLRUnlearner(data, labels, voc, epsilon=1, delta=1, sigma=1, lambda_=1)
    for n_features in combination_lengths:
        avg_affected_rows = 0
        avg_affected_entries = 0
        for _ in tqdm(range(repetitions)):
            choice = np.random.choice(indices_to_delete, n_features, replace=False)
            data_copy, relevant_rows = unlearner.copy_and_replace(data[0], choice, remove=True)
            avg_affected_rows += len(relevant_rows) / repetitions
            avg_affected_entries += ((data[0].nnz-data_copy.nnz)/data[0].nnz) / repetitions
        print(f'{n_features} removed features result in {avg_affected_rows} affected samples on average.')
        print(f'{n_features} removed features result in {avg_affected_entries} affected entries on average.')
        with open(os.path.join(save_path, 'affected_samples'), 'a') as f:
            print(f'{n_features} removed features result in {avg_affected_rows} affected samples on average.', file=f)
            print(f'{n_features} removed features result in {avg_affected_entries} affected entries on average.', file=f)


def runtime_experiments(train_data, test_data, voc, indices_to_delete, combination_length, unlearning_rate,
                        unlearning_rate_ft, n_combinations, remove, save_path, n_replacements, n_shards=5):
    n_shards = 5
    unlearner = DPLRUnlearner(train_data, test_data, voc, epsilon=1, delta=1, sigma=0.1, lambda_=1.0)
    y_train = unlearner.y_train
    name_combinations = [list(np.random.choice(indices_to_delete, combination_length, replace=False)) for _ in
                         range(n_combinations)]
    rt_retrained, rt_first, rt_second, rt_hess, affected, grads_retrained, rt_ft, rt_offset, rt_sisa, grads_sisa = [], [], [], [], [], [], [], [], [], []
    for indices in tqdm(name_combinations):
        # we do not count detection of "broken" features into runtime
        x_delta, changed_rows = unlearner.copy_and_replace(unlearner.x_train, indices, remove, n_replacements=n_replacements)
        x_delta_test, _ = unlearner.copy_and_replace(unlearner.x_test, indices, remove, n_replacements=n_replacements)
        start_time_1 = time.time()
        z = (unlearner.x_train[changed_rows], unlearner.y_train[changed_rows])
        z_delta = (x_delta[changed_rows], unlearner.y_train[changed_rows])
        G = unlearner.get_G(z, z_delta)
        end_time_1 = time.time()
        const_offset = end_time_1 - start_time_1
        train_data_splits, data_indices = split_train_data(n_shards, train_data, indices_to_delete=None,
                                                           remove=remove, n_replacements=n_replacements)
        initial_models = create_models(lambda_=1.0, sigma=0.1, data_splits=train_data_splits,
                                       data_indices=data_indices, test_data=test_data)
        tmp_ensemble = LinearEnsemble(initial_models, n_classes=2)
        tmp_ensemble.update_models(changed_rows)

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
        #t_G = measure_time(unlearner.get_G, [z, z_delta])
        t_ft = measure_time(unlearner.get_fine_tuning_update, [x_delta, y_train, unlearning_rate_ft])
        t_sisa = measure_time(tmp_ensemble.train_ensemble, [])
        grads_sisa.append(tmp_ensemble.get_gradient_calls())

        rt_retrained.append(t_retraining)
        rt_first.append(t_first+const_offset)
        rt_second.append(t_second+const_offset)
        rt_hess.append(t_hess)
        #rt_G.append(t_G)
        rt_ft.append(t_ft)
        rt_offset.append(const_offset)
        rt_sisa.append(t_sisa)
    save_name = os.path.join(save_path, 'runtime_results')
    with open(save_name, 'w') as f:
        print('Avg runtime retraining:', np.mean(rt_retrained), file=f)
        print('Avg runtime first:', np.mean(rt_first), file=f)
        print('Avg runtime second:', np.mean(rt_second), file=f)
        print('Avg runtime hess:', np.mean(rt_hess), file=f)
        print('Avg runtime ft:', np.mean(rt_ft), file=f)
        #print('Avg runtime G:', np.mean(rt_G), file=f)
        print('Avg runtime offset:', np.mean(rt_offset), file=f)
        print('Avg runtime SISA:', np.mean(rt_sisa), file=f)
        print('Avg gradients 1st order:', np.mean(affected), file=f)
        print('Avg gradients retraining:', np.mean(grads_retrained), file=f)
        print('Avg gradients sisa:', np.mean(grads_sisa), file=f)
    print(f'Saved results at {save_name}')


def dnn_training(train_data, test_data, voc, save_folder, n_layers, n_neurons, optimizer, learning_rate, batch_size,
                 epochs, l2_reg):
    if len(train_data[1].shape) == 1:
        train_data = (train_data[0], to_categorical(train_data[1], 2))
        test_data = (test_data[0], to_categorical(test_data[1], 2))

    unlearner = DNNUnlearner(train_data, test_data, test_data, voc, lambda_=l2_reg, n_layers=n_layers,
                            n_neurons=n_neurons, optimizer=optimizer, learning_rate=learning_rate)
    unlearner.train_model(save_folder, batch_size=batch_size, epochs=epochs)


def main(args):
    normalize = args.normalize if hasattr(args, 'normalize') else False
    loader = DataLoader(args.dataset_name, normalize)
    train_data, test_data, voc = (loader.x_train, loader.y_train), (loader.x_test, loader.y_test), loader.voc
    category_to_idx_dict = loader.category_to_idx_dict
    res_save_folder = 'Results_{}'.format(args.dataset_name)
    model_save_folder = 'Models_{}'.format(args.dataset_name)
    if not os.path.isdir(res_save_folder):
        os.makedirs(res_save_folder)
    relevant_features = loader.relevant_features
    relevant_indices = [voc[f] for f in relevant_features]
    if hasattr(args, 'indices_choice'):
        if args.indices_choice == 'all':
            indices_to_delete = list(range(train_data[0].shape[1]))
        elif args.indices_choice == 'relevant':
            indices_to_delete = relevant_indices
        elif args.indices_choice == 'most_important':
            indices_to_delete, _ = find_most_relevant_indices(train_data, test_data, voc, top_n=args.most_important_size)
        elif args.indices_choice == 'intersection':
            # intersection
            top_indices, _ = find_most_relevant_indices(train_data, test_data, voc, top_n=args.most_important_size)
            indices_to_delete = np.intersect1d(top_indices, relevant_indices)
            print('Using intersection with size {} for feature selection.'.format(len(indices_to_delete)))
        elif args.indices_choice == 'union':
            top_indices, _ = find_most_relevant_indices(train_data, test_data, voc, top_n=args.most_important_size)
            indices_to_delete = np.union1d(top_indices, relevant_indices)
            print('Using union with size {} for feature selection.'.format(len(indices_to_delete)))
    if args.command == 'sigma-performance':
        sigma_performance_experiments(train_data, test_data, voc, args.lambdas, args.sigmas, args.repetitions, res_save_folder)
    elif args.command == 'avg-grad-residual':
        get_average_gradient_residual(train_data, test_data, voc, args.Lambda, args.sigma, indices_to_delete,
                                      args.combination_lengths, args.n_combinations, args.unlearning_rate,
                                      args.unlearning_rate_ft, args.iter_steps, category_to_idx_dict, remove=args.remove,
                                      n_replacements=args.n_replacements, save_path=res_save_folder)
    elif args.command == 'scatter':
        scatter_experiments(train_data, test_data, voc, args.Lambda, args.sigma, indices_to_delete,
                            args.combination_length, args.unlearning_rate, args.unlearning_rate_ft,
                            args.n_combinations, category_to_idx_dict, args.remove, res_save_folder, args.n_replacements)
    elif args.command == 'fidelity':
        fidelity_experiments(train_data, test_data, voc, args.Lambda, args.sigma, indices_to_delete,
                             args.combination_length, args.unlearning_rate, args.unlearning_rate_ft, args.iter_steps,
                             args.n_combinations, args.remove, res_save_folder, args.n_replacements, args.n_shards,
                             normalize, category_to_idx_dict)
    elif args.command == 'affected_samples':
        get_affected_samples(train_data, test_data, voc, indices_to_delete, args.combination_lengths,
                             args.repetitions, res_save_folder)
    elif args.command == 'runtime':
        runtime_experiments(train_data, test_data, voc, indices_to_delete, args.combination_length,
                            args.unlearning_rate, args.unlearning_rate_ft, args.n_combinations, args.remove,
                            res_save_folder, args.n_replacements)
    elif args.command == 'dnn_training':
        dnn_training(train_data, test_data, voc, model_save_folder, args.n_layers, args.n_neurons, args.optimizer,
                     args.learning_rate, args.batch_size, args.epochs, args.l2_reg)
    else:
        print('Argument not understood')


if __name__ == '__main__':
    experiment_choices = ['sigma_performance', 'avg_grad_res', 'scatter_loss', 'affected_samples', 'fidelity', 'runtime']
    indices_choices = ['all', 'most_important', 'relevant', 'intersection', 'union']
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
    avg_grad_parser.add_argument('n_combinations', type=int, help="How many combinations of features to sample")
    avg_grad_parser.add_argument('unlearning_rate', type=float, help="Unlearning rate for first order update.")
    avg_grad_parser.add_argument('unlearning_rate_ft', type=float, help="Unlearning rate for fine tuning.")
    avg_grad_parser.add_argument('iter_steps', type=int, help="Number of steps for iterative updates.")
    avg_grad_parser.add_argument('--combination_lengths', nargs='*', type=int, help="How many features to choose.")
    avg_grad_parser.add_argument('--remove', action='store_true', help='if set features will be removed entirely.')
    avg_grad_parser.add_argument('--indices_choice', type=str, choices=indices_choices, default='all',
                                help='How to select indices for unlearning.')
    avg_grad_parser.add_argument('--most_important_size', type=int, default=10,
                                 help='Number of most important features to use.'
                                      'Will not be used if indices_choice != most_important or intersection')
    avg_grad_parser.add_argument('--n_replacements', type=int, nargs='*', default=[10], help="If remove==False this number selects"
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
    fidelity_parser.add_argument('iter_steps', type=int, help="Number of steps for iterative updates.")
    fidelity_parser.add_argument('--indices_choice', type=str, choices=indices_choices, default='all',
                                help='How to select indices for unlearning.')
    fidelity_parser.add_argument('--most_important_size', type=int, default=10,
                                 help='Number of most important features to use.'
                                      'Will not be used if indices_choice != most_important or intersection')
    fidelity_parser.add_argument('--remove', action='store_true', help='if set features will be removed entirely.')
    fidelity_parser.add_argument('--normalize', action='store_true', help='if set features will be l2 normalized.')
    fidelity_parser.add_argument('--n_replacements', type=int, nargs='*', default=[0], help="If remove==False this number selects"
                                                                               "the number of samples that will be selected for change.")
    fidelity_parser.add_argument('--n_shards', type=int, help='Number of shards', default=20)
    # affected samples
    impact_parser = subparsers.add_parser("affected_samples", help="Compute how many samples are affected by deletion")
    impact_parser.add_argument('dataset_name', type=str, choices=['Enron', 'Adult', 'Diabetis', 'Drebin'])
    impact_parser.add_argument('repetitions', type=int, help="How often to sample combinations in each run")
    impact_parser.add_argument('--combination_lengths', type=int, nargs='*',
                               help="How many features to choose in each run")
    impact_parser.add_argument('--indices_choice', type=str, choices=indices_choices, default='all',
                                help='How to select indices for unlearning.')
    impact_parser.add_argument('--most_important_size', type=int, default=10,
                                 help='Number of most important features to use.'
                                      'Will not be used if indices_choice != most_important or intersection')
    # runtime parser
    runtime_parser = subparsers.add_parser("runtime", help="Compute average runtime of methods")
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
    # additional experiments with DNNs instead of LR
    dnn_train_parser = subparsers.add_parser("dnn_training", help="Train DNN models for models")
    dnn_train_parser.add_argument('dataset_name', type=str, choices=['Enron', 'Adult', 'Diabetis', 'Drebin'])
    dnn_train_parser.add_argument('--n_layers', type=int, default=2)
    dnn_train_parser.add_argument('--n_neurons', type=int, default=100)
    dnn_train_parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam')
    dnn_train_parser.add_argument('--learning_rate', type=float, default=1e-3)
    dnn_train_parser.add_argument('--l2_reg', type=float, default=1e-3)
    dnn_train_parser.add_argument('--batch_size', type=int, default=64)
    dnn_train_parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    main(args)
