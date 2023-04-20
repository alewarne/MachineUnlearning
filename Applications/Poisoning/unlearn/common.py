import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical

from util import LoggedGradientTape, ModelTmpState, CSVLogger, measure_time, GradientLoggingContext
from Applications.Poisoning.unlearn.core import approx_retraining


def evaluate_model_diff(model, new_model, x_valid, y_valid, diverged=False, verbose=False, ref_acc=0.87):

    acc_before_fix = model.evaluate(x_valid, y_valid, verbose=0)[1]
    acc_after_fix = -1
    if not diverged:
        acc_after_fix = new_model.evaluate(x_valid, y_valid, verbose=0)[1]
    if verbose:
        acc_restored = (acc_after_fix - acc_before_fix) / (ref_acc - acc_before_fix)
        print(f">> acc_restored={acc_restored}, acc_before={acc_before_fix}, "
              f"acc_after={acc_after_fix}, lissa diverged: {diverged}")
    return acc_before_fix, acc_after_fix, diverged


def evaluate_unlearning(model_init, model_weights, data, delta_idx, y_train_orig, unlearn_kwargs, repaired_filepath=None,
                        clean_acc=1.0, verbose=False, cm_dir=None, log_dir=None):
    clear_session()
    (x_train, y_train), _, (x_valid, y_valid) = data
    model = model_init()
    params = np.sum(np.product([xi for xi in x.shape]) for x in model.trainable_variables).item()
    model.load_weights(model_weights)
    new_theta, diverged, logs, duration_s = unlearn_update(
        x_train, y_train, y_train_orig, delta_idx, model, x_valid, y_valid, unlearn_kwargs, verbose=verbose, cm_dir=cm_dir, log_dir=log_dir)

    new_model = model_init()
    new_model.set_weights(new_theta)
    if repaired_filepath is not None:
        new_model.save_weights(repaired_filepath)

    acc_before, acc_after, diverged = evaluate_model_diff(
        model, new_model, x_valid, y_valid, diverged, verbose, clean_acc)
    return acc_before, acc_after, diverged, logs, duration_s, params


def unlearn_update(z_x, z_y, z_y_delta, delta_idx, model, x_val, y_val, unlearn_kwargs,
                   verbose=False, cm_dir=None, log_dir=None):
    assert np.min(delta_idx) >= 0 and np.max(delta_idx) < z_x.shape[0]

    z_x = tf.constant(z_x, dtype=tf.float32)
    z_y_delta = tf.constant(z_y_delta, dtype=tf.int32)
    with GradientLoggingContext('unlearn'):
        new_theta, diverged, duration_s = iter_approx_retraining(z_x, z_y_delta, model, x_val, y_val, delta_idx, verbose=verbose,
                                                                 cm_dir=cm_dir, log_dir=log_dir, **unlearn_kwargs)
    return new_theta, diverged, LoggedGradientTape.logs['unlearn'], duration_s


def iter_approx_retraining(z_x, z_y_delta, model, x_val, y_val, delta_idx, max_inner_steps=1,
                           steps=1, verbose=False, cm_dir=None, log_dir=None, **unlearn_kwargs):
    """Iterative approximate retraining.

    Args:
        z_x (np.ndarray): Original features.
        z_y (np.ndarray): Original labels.
        z_x_delta (np.ndarray): Changed features.
        z_y_delta (np.ndarray): Changed labels.
        delta_idx (np.ndarray): Indices of the data to change.
        steps (int, optional): Number of iterations. Defaults to 1.
        mixing_ratio (float, optional): Ratio of unchanged data to mix in. Defaults to 1.
        cm_dir (str, optional): If provided, plots confusion matrices afrer each iterations into this directory.
                                Defaults to None.
        verbose (bool, optional): Verbosity switch. Defaults to False.

    Returns:
        list: updated model parameters
        bool: whether the LiSSA algorithm diverged
    """

    # take HVP batch size from kwargs
    hvp_batch_size = unlearn_kwargs.get('hvp_batch_size', 512)

    # setup loggers
    if log_dir is None:
        step_logger, batch_logger, hvp_logger = None, None, None
    else:
        step_logger = CSVLogger('step', ['step', 'batch_acc', 'val_acc', 'delta_size',
                                         'new_errors', 'remaining_delta'], os.path.join(log_dir, 'log_step.csv'))
        batch_logger = CSVLogger('batch', ['step', 'inner_step', 'batch_acc'], os.path.join(log_dir, 'log_batch.csv'))
        hvp_logger = CSVLogger('hvp', ['step', 'inner_step', 'i', 'update_norm'], os.path.join(log_dir, 'log_hvp.csv'))

    model_weights = model.get_weights()
    analysis_time = 0  # allow for additional (slow) analysis code that is not related to the algorithm itself
    # the TmpState context managers restore the states of weights, z_x, z_y, ... afterwards
    with measure_time() as total_timer, ModelTmpState(model):
        idx, prio_idx = get_delta_idx(model, z_x, z_y_delta, hvp_batch_size)
        batch_acc_before = 0.0
        for step in range(0, steps+1):
            with measure_time() as t:
                val_acc_before = model.evaluate(x_val, y_val, verbose=0)[1]
                analysis_time += t()
            if step == 0:
                # calc initial metrics in step 0
                batch_acc_after = batch_acc_before
                val_acc_after = val_acc_before
            else:
                # fixed arrays during unlearning
                _z_x = tf.gather(z_x, idx)
                _z_x_delta = tf.identity(_z_x)
                _z_y_delta = tf.gather(z_y_delta, idx)

                for istep in range(1, max_inner_steps+1):
                    hvp_logger.step = step
                    hvp_logger.inner_step = istep
                    # update model prediction after each model update
                    z_y_pred = to_categorical(np.argmax(batch_pred(model, _z_x), axis=1), num_classes=10)
                    new_theta, diverged = approx_retraining(model, _z_x, z_y_pred, _z_x_delta, _z_y_delta,
                                                            hvp_x=z_x, hvp_y=z_y_delta, hvp_logger=hvp_logger, **unlearn_kwargs)
                    # don't update if the LiSSA algorithm diverged
                    if diverged:
                        break

                    # update weights
                    model_weights[-len(new_theta):] = new_theta
                    model.set_weights(model_weights)

                    batch_acc_after = model.evaluate(_z_x, _z_y_delta, verbose=0)[1]
                    if verbose:
                        print(f"> {istep}: batch_acc = {batch_acc_after}")
                    if batch_logger is not None:
                        batch_logger.log(step=step, inner_step=istep, batch_acc=batch_acc_after)
                    if batch_acc_after == 1.0:
                        break
                with measure_time() as t:
                    val_acc_after = model.evaluate(x_val, y_val, verbose=0)[1]
                    analysis_time += t()

            # get index of next delta set
            idx, prio_idx = get_delta_idx(model, z_x, z_y_delta, hvp_batch_size)
            with measure_time() as t:
                if step_logger is not None:
                    new_errors = len(set(prio_idx) - set(delta_idx))
                    remaining_delta = len(set(prio_idx) & set(delta_idx))
                    step_logger.log(step=step, batch_acc=batch_acc_after, val_acc=val_acc_after,
                                    delta_size=len(prio_idx), new_errors=new_errors, remaining_delta=remaining_delta)
                if verbose:
                    print(f">> iterative approx retraining ({len(idx)} samples): step = {step}, train_acc (before/after) = {batch_acc_before} / {batch_acc_after}, "
                        f"val_acc = {val_acc_before} / {val_acc_after}")
                if cm_dir is not None:
                    title = f'After Unlearning Step {step}' if step > 0 else 'Before Unlearning'
                    plot_cm(x_val, y_val, model, title=title,
                            outfile=os.path.join(cm_dir, f'cm_unlearning_{step:02d}.png'))
                analysis_time += t()

        duration_s = total_timer() - analysis_time
    return model_weights, diverged, duration_s


def get_delta_idx(model, x, y, batch_size):
    y_pred = np.argmax(batch_pred(model, x), axis=1)
    prio_idx = np.argwhere(y_pred != np.argmax(y, axis=1))[:, 0]
    idx = np.random.choice(prio_idx, min(batch_size, len(prio_idx)), replace=False)
    return idx, prio_idx


def get_mixed_delta_idx(delta_idx, n_samples, mixing_ratio=1.0, prio_idx=None):
    """Mix regular training data into delta set.

    Args:
        delta_idx (np.ndarray): Indices of the data to unlearn.
        n_samples (int): Total number of samples.
        mixing_ratio (float, optional): Ratio of regular data points to mix in. Defaults to 1.0.
        prio_idx (np.ndarray, optional): Indices of training samples to prioritize during unlearning.
                                                Defaults to None.

    Returns:
        np.ndarray: Indeces of delta samples with added regular data.
    """
    if mixing_ratio == 0.0:
        return delta_idx

    priority_idx = list(set(prio_idx) - set(delta_idx)) if prio_idx is not None else []
    if mixing_ratio == -1:
        return np.hstack((delta_idx, priority_idx)).astype(np.int)

    remaining_idx = list(set(range(n_samples)) - set(delta_idx) - set(priority_idx))
    n_total = np.ceil(mixing_ratio*delta_idx.shape[0]).astype(np.int) + delta_idx.shape[0]
    n_prio = min(n_total, len(priority_idx))
    n_regular = max(n_total - len(priority_idx) - len(delta_idx), 0)
    idx = np.hstack((
        delta_idx,
        np.random.choice(priority_idx, n_prio, replace=False),
        np.random.choice(remaining_idx, n_regular, replace=False)))
    return idx.astype(np.int)


def plot_cm(x, y_true, model, title='confusion matrix', outfile=None):
    y_pred = np.argmax(batch_pred(model, x), axis=1)
    y_true = np.argmax(y_true, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    df_cm = pd.DataFrame(cm, range(n_classes), range(n_classes))
    sns.set(font_scale=1.4)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g', ax=ax, cbar=False)
    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile, dpi=300)


def batch_pred(model, x, batch_size=2048):
    preds = []
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        preds.append(model(x[start:end]))
    return tf.concat(preds, 0)
