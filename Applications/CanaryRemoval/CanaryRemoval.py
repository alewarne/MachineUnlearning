import sys
import os

import numpy as np
import json
import argparse
from .AlicePreprocessing import load_data
import time

sys.path.append('../../')

from Unlearner.RNNUnlearner import RNNUNlearner

CANARY_STR = "`my telephone number is {}!' said alice.\n\n  "
CANARY_START = "`my telephone number is "


# creates training examples and labels with random numbers to delete bias of the single number
def get_z_delta(x_train_old, data_path, canary_no, seq_length, idx2char, n_repetitions, replace_char):
    card_no_length = len(canary_no)
    # if single char is given we simply repeat it
    if len(replace_char) == 1:
        fixed_number = replace_char*card_no_length
    # else we use the given string
    else:
        assert len(replace_char) == card_no_length
        fixed_number = replace_char
    fixed_canary = CANARY_STR.format(fixed_number)
    x_train_new, y_train_new, idx2char_new = load_data(data_path, seq_length, fixed_canary, n_repetitions, idx2char)
    assert idx2char == idx2char_new
    # check where the datasets are different -> only at the points where the random numbers appear
    affected_indices = np.unique(np.where(x_train_old != x_train_new)[0])
    x_delta = x_train_new[affected_indices]
    y_delta = y_train_new[affected_indices]
    print('{} of {} samples are affected by the unlearning ({}%)'.format(x_delta.shape[0], x_train_old.shape[0],
                                                                         (100 * x_delta.shape[0] / x_train_old.shape[0])
                                                                         ))
    return affected_indices, x_delta, y_delta


def unlearn_canary(unlearner, data_path, seq_length, n_canaries, tau, order, batch_size, scale, damping, iterations,
                   replace_char, rounds=1, train_reduction=1.0, epochs=1, eval_reduction=None, stabilization_epochs=0,
                   mixing_ratio=1.0, verbose=False):
    chars_to_predict = 80
    if verbose:
        print('Testing canary before unlearning step ...')
        pp_start, loss_start, acc_start, _ = unlearner.test_canary(reference_char=replace_char,
                                                                   chars_to_predict=chars_to_predict,
                                                                   train_reduction=eval_reduction)
    else:
        pp_start, loss_start, acc_start = -1, -1, -1
    indices_to_change, x_delta, y_delta = get_z_delta(unlearner.x_train, data_path, unlearner.canary_number, seq_length,
                                                      unlearner.idx2char, n_canaries, replace_char)
    if train_reduction != 1:
        x_train_old = unlearner.x_train.copy()
        y_train_old = unlearner.y_train.copy()
        z_x_old, z_y_old = unlearner.x_train[indices_to_change].copy(), unlearner.y_train[indices_to_change].copy()
        idx_train_2_idx_delta = {i: j for i, j in zip(indices_to_change, range(x_delta.shape[0]))}
        unlearner.reduce_train_set(train_reduction, delta_idx=indices_to_change)
        # map the indices that were chosen back to the indices of x_delta
        indices_delta_reduced = np.array([idx_train_2_idx_delta[idx] for idx in
                                 unlearner.new_train_indices[unlearner.delta_idx_train]])
        z_x_reduced = z_x_old[indices_delta_reduced]
        z_y_reduced = z_y_old[indices_delta_reduced]
        z_x_delta_reduced = x_delta[indices_delta_reduced]
        z_y_delta_reduced = y_delta[indices_delta_reduced]
        unlearner.update_influence_variables_samples(z_x_reduced, z_y_reduced, z_x_delta_reduced, z_y_delta_reduced)
        x_fixed, y_fixed = unlearner.x_train.copy(), unlearner.y_train.copy()
        x_fixed[unlearner.delta_idx_train] = z_x_delta_reduced
        y_fixed[unlearner.delta_idx_train] = z_y_delta_reduced
    else:
        unlearner.update_influence_variables_samples_indices(indices_to_change, x_delta, y_delta)
        x_fixed, y_fixed = unlearner.x_train.copy(), unlearner.y_train.copy()
        x_fixed[indices_to_change] = x_delta
        y_fixed[indices_to_change] = y_delta
    start_time = time.time()
    if order > 0:
        theta_updated, diverged = unlearner.approx_retraining(hvp_x=x_fixed, hvp_y=y_fixed, batch_size=batch_size,
                                                              scale=scale,
                                                              damping=damping, iterations=iterations, verbose=verbose,
                                                              rounds=rounds, tau=tau, order=order)
        if stabilization_epochs > 0:
            assert not diverged
            unlearner.test_canary(reference_char=replace_char, weights=theta_updated,
                                  chars_to_predict=chars_to_predict,
                                  train_reduction=eval_reduction)
            unlearner.model.set_weights(theta_updated)
            theta_updated, diverged = unlearner.iter_approx_retraining(unlearner.x_train, unlearner.y_train,
                                                                       x_fixed, y_fixed, indices_to_change,
                                                                       prioritize_misclassified=True,
                                                                       steps=stabilization_epochs,
                                                                       verbose=False,
                                                                       batch_size=batch_size, scale=scale,
                                                                       damping=damping, iterations=iterations,
                                                                       rounds=rounds, tau=tau, order=order,
                                                                       mixing_ratio=mixing_ratio)
    else:
        theta_updated = unlearner.fine_tune(x_fixed, y_fixed, learning_rate=tau, batch_size=batch_size, epochs=epochs)
        diverged = False
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Unlearning took {total_time} seconds.')
    if train_reduction != 1:
        unlearner.reduce_train_set(x_train_old=x_train_old, y_train_old=y_train_old)
    pp_end, loss_end, acc_end, completion = unlearner.test_canary(reference_char=replace_char, weights=theta_updated,
                                                                  chars_to_predict=chars_to_predict,
                                                                  train_reduction=eval_reduction)
    return theta_updated, pp_start, pp_end, loss_start, loss_end, acc_start, acc_end, diverged, completion, total_time


# perform single gradient-descent-step based on x and y
def gradient_step(unlearner, x, y, learning_rate):
    model_params = unlearner.model.get_weights()
    grads = unlearner.get_gradients(x,y)
    new_weights = [w-learning_rate*g for w, g in zip(model_params, grads)]
    return new_weights


def get_params_by_model_name(weight_path):
    param_separator = '-'
    value_separator = '='
    filename = os.path.basename(weight_path)
    if filename.endswith('hdf5'):
        param_str = filename.split('.hdf5')[0]
    else:
        param_str = filename.split('.ckpt')[0]
    splits_params = param_str.split(param_separator)
    splits_values = [s.split(value_separator) for s in splits_params]
    lambda_ = float(splits_values[0][1])
    canary_number = splits_values[1][1]
    canary_reps = int(splits_values[2][1])
    embedding_dim = int(splits_values[3][1])
    seq_len = int(splits_values[4][1])
    p_dropout = float(splits_values[5][1]) if len(splits_values) > 5 else 0
    return lambda_, canary_number, canary_reps, embedding_dim, seq_len, p_dropout


def args_check_canary_by_name(args):
    weight_paths, data_path = args.weight_path, args.data_path
    for wp in weight_paths:
        print('Checking {}'.format(wp))
        lambda_, canary_number, canary_reps, embedding_dim, seq_length, p_dropout = get_params_by_model_name(wp)
        canary = CANARY_STR.format(canary_number)
        x_train, y_train, idx2char = load_data(data_path, seq_length, canary, canary_reps)
        unlearner = RNNUNlearner(x_train, y_train, embedding_dim, idx2char, lambda_, wp, CANARY_START, canary_number,
                                 canary_reps, n_layers=args.n_layers, n_units=args.n_units,p_dropout=p_dropout)
        unlearner.generate_data(start_str=args.start_str)


def args_unlearn_canary(args):
    weight_path = args.weight_path
    data_path = args.data_path
    replacement_char = args.replace_char
    batch_size, scale, damping, rounds, verbose = args.batch_size, args.scale, args.damping, args.rounds, args.verbose
    iterations = args.iterations
    stabilization_epochs, mixing_ratio = args.stabilization_epochs, args.mixing_ratio
    lambda_, canary_number, canary_reps, embedding_dim, seq_length, p_dropout = get_params_by_model_name(weight_path)
    tau, order, epochs = args.tau, args.order, args.epochs
    reduction, eval_reduction = args.reduction, args.eval_reduction
    canary = CANARY_STR.format(canary_number)
    x_train, y_train, idx2char = load_data(data_path, seq_length, canary, canary_reps)
    initial_perplexities_path, perplexity_samples = args.initial_perplexities_path, args.perplexity_samples
    calc_exposures, recalc_exposures, digits_only = args.calc_exposures, args.recalc_exposures, args.digits_only
    unlearner = RNNUNlearner(x_train, y_train, embedding_dim, idx2char, lambda_, weight_path, CANARY_START,
                             canary_number, canary_reps, n_layers=args.n_layers, n_units=args.n_units,
                             p_dropout=p_dropout)
    if calc_exposures:
        if initial_perplexities_path is not None:
            initial_perplexities = np.load(initial_perplexities_path)
        else:
            print('Calculating initial perplexity distribution ...')
            initial_perplexities = unlearner.calc_perplexity_distribution(no_samples=perplexity_samples,
                                                                          only_digits=digits_only)

    print(unlearner.model.summary())
    tup = unlearn_canary(unlearner, data_path, seq_length, canary_reps, tau, order, batch_size, scale, damping,
                         iterations, replacement_char, rounds, reduction, epochs, eval_reduction, stabilization_epochs,
                         mixing_ratio, verbose)
    theta_updated = tup[0]
    pp_start, pp_end = tup[1], tup[2]
    acc_start, acc_end = tup[5], tup[6]
    completion = tup[8]
    total_time = tup[9]
    res_dict = {'acc_start': acc_start, 'acc_end': acc_end, 'perplexity_start': pp_start, 'perplexity_end': pp_end,
                'completion': completion, 'time': total_time}
    if calc_exposures:
        # this is expensive. usually it is enough to use the initial perplexities
        if recalc_exposures:
            print('Approximating exposure before unlearning ...')
            if pp_start == -1:
                initial_exposure = 0
            else:
                initial_exposure = unlearner.approx_exposure(pp_start, initial_perplexities, only_digits=digits_only)[0]
            print('Calculating perplexity distribution after learning')
            unlearned_perplexities = unlearner.calc_perplexity_distribution(theta_updated, perplexity_samples,
                                                                            only_digits=digits_only)
            final_exposure = unlearner.approx_exposure(pp_end, unlearned_perplexities, only_digits=digits_only)[0]
        else:
            if pp_start == -1:
                initial_exposure = 0
                final_exposure = unlearner.approx_exposure(pp_end, initial_perplexities, only_digits=digits_only)[0]
            else:
                # try to call this method only once to save time
                initial_exposure, final_exposure = unlearner.approx_exposure([pp_start, pp_end], initial_perplexities,
                                                                             only_digits=digits_only)
        res_dict['exposure_start'] = initial_exposure
        res_dict['exposure_end'] = final_exposure
    unlearned_model = unlearner.get_network(no_lstm_units=args.n_units, n_layers=args.n_layers)
    unlearned_model.set_weights(theta_updated)
    if order == 0:
        save_name_model_f = 'finetuned_{}-replacement={}-reduction={}-learning_rate={}-epochs={}-batch_size={}.hdf5'
        save_name_log_f = 'finetuned_{}-replacement={}-reduction={}-learning_rate={}-epochs={}-batch_size={}.json'
        save_name_model = save_name_model_f.format(unlearner.param_string, replacement_char, reduction, tau, epochs,
                                                   batch_size)
        save_name_log = save_name_log_f.format(unlearner.param_string, replacement_char, reduction, tau, epochs,
                                               batch_size)
    elif order == 1:
        save_name_model_f = 'unlearnedLinear_{}-replacement={}-reduction={}-learning_rate={}-epochs={}.hdf5'
        save_name_log_f = 'unlearnedLinear_{}-replacement={}-reduction={}-learning_rate={}-epochs={}.json'
        save_name_model = save_name_model_f.format(unlearner.param_string, replacement_char, reduction, tau, epochs)
        save_name_log = save_name_log_f.format(unlearner.param_string, replacement_char, reduction, tau, epochs)
    else:
        save_name_model_f = 'unlearnedSecond_{}-replacement={}-reduction={}-learning_rate={}-epochs={}-damping={}-' \
                            'scale={}.hdf5'
        save_name_log_f = 'unlearnedSecond_{}-replacement={}-reduction={}-learning_rate={}-epochs={}-damping={}-' \
                          'scale={}.json'
        save_name_model = save_name_model_f.format(unlearner.param_string, replacement_char, reduction, tau, epochs,
                                                   damping, scale)
        save_name_log = save_name_log_f.format(unlearner.param_string, replacement_char, reduction, tau, epochs,
                                               damping, scale)
    if args.save_model:
        save_path_model = os.path.join(args.save_path, save_name_model)
        unlearned_model.save_weights(save_path_model)
        print('Saved unlearned model at {}'.format(save_path_model))
    save_path_log = os.path.join(args.save_path, save_name_log)
    json.dump(res_dict, open(save_path_log, 'w'), indent=4)
    print('Saved log at {}'.format(save_path_log))


def calc_perplexity(args):
    data_path = args.data_path
    weight_path = args.weight_path
    lambda_, canary_number, canary_reps, embedding_dim, seq_length, p_dropout = get_params_by_model_name(weight_path)
    canary = CANARY_STR.format(canary_number)
    x_train, y_train, idx2char = load_data(data_path, seq_length, canary, canary_reps)
    unlearner = RNNUNlearner(x_train, y_train, embedding_dim, idx2char, lambda_, weight_path, CANARY_START, canary_number,
                             canary_reps, n_layers=args.n_layers, n_units=args.n_units, p_dropout=p_dropout)
    perplexities = unlearner.calc_perplexity_distribution(no_samples=args.n_samples, plot=args.plot,
                                                          only_digits=args.digits_only)
    save_path = args.save_path if args.save_path is not None else '.'
    filename_fstr = 'perplexities_samples-{}_canary_no={}_n_layers={}_n_units={}_lambda={}_canary_reps={}_dropout={}.npy'
    filename = filename_fstr.format(args.n_samples, canary_number, args.n_layers, args.n_units, lambda_, canary_reps,
                                    p_dropout)
    filepath = os.path.join(save_path, filename)
    np.save(filepath, perplexities)
    print('Saved perplexities at {}'.format(filepath))


def approx_exposure(args):
    weight_path = args.weight_path
    data_path = args.data_path
    lambda_, canary_number, canary_reps, embedding_dim, seq_length, p_dropout = get_params_by_model_name(weight_path)
    canary = CANARY_STR.format(canary_number)
    x_train, y_train, idx2char = load_data(data_path, seq_length, canary, canary_reps)
    unlearner = RNNUNlearner(x_train, y_train, embedding_dim, idx2char, lambda_, weight_path,
                             CANARY_START, canary_number, canary_reps, n_layers=args.n_layers, n_units=args.n_units,
                             p_dropout=p_dropout)
    perplexities = np.load(args.perplexity_path)
    unlearner.approx_exposure(args.perplexity, perplexities, only_digits=args.digits_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    # check canary model
    canary_check_parser = subparsers.add_parser("check_canary", help="Predict canary for models.")
    canary_check_parser.add_argument("weight_path", type=str, nargs='*', help='Path to model(s) checkpoint(s).')
    canary_check_parser.add_argument("data_path", type=str, help='Path to training data.')
    canary_check_parser.add_argument("--n_layers", type=int, help="No lstm layers.", default=1)
    canary_check_parser.add_argument("--n_units", type=int, help="No lstm units.", default=256)
    canary_check_parser.add_argument("--start_str", type=str, help="Start string to predict from")
    # remove canary with approximate method
    unlearn_parser = subparsers.add_parser("unlearn", help="Unlearn credit card via influence functions.")
    unlearn_parser.add_argument("weight_path", type=str, help='Path to model checkpoint.')
    unlearn_parser.add_argument("data_path", type=str, help='Path to training data.')
    unlearn_parser.add_argument("replace_char", type=str, help='Char to replace credit-card-number with.')
    unlearn_parser.add_argument("--order", type=int, help="Method for deletion. 0=Fine-tuning with SGD, 1=First order"
                                                          "2 = second order removal", default=1, choices=[0, 1, 2])
    # unlearning parameters
    unlearn_parser.add_argument("--scale", type=float, help='Scaling factor of loss function.', default=1e4)
    unlearn_parser.add_argument("--damping", type=float, help='Dampig value added to Hessian.', default=0.1)
    unlearn_parser.add_argument("--batch_size", type=int, default=500, help='Order 0: Batch size of fine-tuning batches.'
                                                               'Order 2: Size of batches used in LISSA algorithm.')
    unlearn_parser.add_argument("--tau", type=float, help="Order 0: Learning rate; Order1: unlearning rate.",
                                default=0.0001)
    unlearn_parser.add_argument("--iterations", type=int, help='Iterations of batch sampling in LISSA.', default=-1)
    unlearn_parser.add_argument("--epochs", type=int, help='Epochs of fine-tuning for order 0.', default=1)
    unlearn_parser.add_argument("--rounds", type=int, help='Repetitions of LISSA.', default=1)
    unlearn_parser.add_argument("--verbose", action='store_true', help='Verbose output of LISSA convergence')
    unlearn_parser.add_argument("--n_layers", type=int, help="No lstm layers.", default=1)
    unlearn_parser.add_argument("--n_units", type=int, help="No lstm units.", default=256)
    unlearn_parser.add_argument("--reduction", type=float, help="Reduction of training data and affected samples",
                                default=1.0)
    unlearn_parser.add_argument("--stabilization_epochs", type=int, help="Epochs of stabilization", default=0)
    unlearn_parser.add_argument("--mixing_ratio", type=float, help="Ratio of incorrect points to add with stabilization",
                                default=1.0)
    unlearn_parser.add_argument("--eval_reduction", type=int, help="Calculation of accuracy will be performed only on"
                                "this number of samples. When using Second-order methods GPU is not available and"
                                "the evaluation on CPU takes long. In this case this argument is useful.")
    # exposure calculation
    unlearn_parser.add_argument("--initial_perplexities_path", type=str, help='Path to initial perplexities to compute'
                                                                              'exposure.')
    unlearn_parser.add_argument("--perplexity_samples", type=int, help='No samples to approx exposure.',
                                default=10000000)
    unlearn_parser.add_argument("--calc_exposures", help="Whether or not to calc exposure at all", action="store_true")
    unlearn_parser.add_argument("--recalc_exposures", help="Whether or not to calculate exposure after unlearning for"
                                "_new_ corresponding model. Normally the difference to the original model is very low"
                                "and it is sufficient to use initial perplexity distribution", action="store_true")
    unlearn_parser.add_argument("--digits_only", help="Whether to save the model emerging from unlearning",
                                action="store_true")
    unlearn_parser.add_argument("--save_path", type=str, help='Where to save unlearned model weights and report.',
                                default='.')
    unlearn_parser.add_argument("--save_model", help="Whether to save the model emerging from unlearning",
                                action="store_true")
    # calc perplexity
    perplexity_parser = subparsers.add_parser("calc_perplexity", help="Calc and save perplexity distribution.")
    perplexity_parser.add_argument("weight_path", type=str, help='Path to model checkpoint.')
    perplexity_parser.add_argument("data_path", type=str, help='Path to training data.')
    perplexity_parser.add_argument("--n_samples", type=int, help='How many sequences to sample', default=1000000)
    perplexity_parser.add_argument("--n_layers", type=int, help="No lstm layers.", default=1)
    perplexity_parser.add_argument("--n_units", type=int, help="No lstm units.", default=256)
    perplexity_parser.add_argument("--save_path", type=str, help='Where to save exposures.')
    perplexity_parser.add_argument("--digits_only", action='store_true',
                                   help='Whether to use only digits for distribution.')
    perplexity_parser.add_argument("--plot", action='store_true', help='Whether to show histogram plot.')
    # approx exposure with perplexity distribution
    exposure_parser = subparsers.add_parser("approx_exposure", help="Approx rank of given perplexity via density.")
    exposure_parser.add_argument("weight_path", type=str, help='Path to model checkpoint.')
    exposure_parser.add_argument("data_path", type=str, help='Path to training data.')
    exposure_parser.add_argument("perplexity_path", type=str, help='Path to all perplexities (.npy).')
    exposure_parser.add_argument("perplexity", type=float, help='Perplexity to calculate rank for.')
    exposure_parser.add_argument("--n_layers", type=int, help="No lstm layers.", default=1)
    exposure_parser.add_argument("--n_units", type=int, help="No lstm units.", default=256)
    exposure_parser.add_argument("--digits_only", action='store_true',
                                 help='Whether to use only digits for distribution.')

    args = parser.parse_args()
    if args.command == "check_canary":
        args_check_canary_by_name(args)
    elif args.command == "unlearn":
        args_unlearn_canary(args)
    elif args.command == "calc_perplexity":
        calc_perplexity(args)
    elif args.command == "approx_exposure":
        approx_exposure(args)
