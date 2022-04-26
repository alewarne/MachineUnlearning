import numpy as np
import json
import tensorflow as tf
import os
import hashlib
import scipy.sparse as sp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from Unlearner.DrebinDataGenerator import DrebinDataGenerator
from scipy.optimize import fmin_ncg
from scipy.sparse.linalg import eigsh, LinearOperator

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

from util import TrainingResult, save_train_results, ModelTmpState, DeltaTmpState, LoggedGradientTape


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 130:
        lr *= 0.5e-3
    elif epoch > 110:
        lr *= 1e-3
    elif epoch > 90:
        lr *= 1e-2
    elif epoch > 50:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


class DNNUnlearner:
    def __init__(self, train, test, valid, voc, weight_path=None, class_weights=False, lambda_=1e-3):
        tf.random.set_seed(42)
        assert train[0].shape[0] == train[1].shape[0]
        self.x_train = train[0]
        self.y_train = train[1]
        self.x_test = test[0]
        self.y_test = test[1]
        self.x_valid = valid[0]
        self.y_valid = valid[1]
        self.voc = voc
        self.lambda_ = lambda_
        self.n = self.x_train.shape[0]
        self.dim = self.x_train.shape[1]
        self.model = self.get_network(weight_path)
        if class_weights:
            class_labels = np.argmax(self.y_train, axis=1) if len(self.y_train.shape) > 1 else self.y_train
            neg, pos = np.bincount(class_labels)
            total = neg+pos
            # the following is taken from tensorflow documentation
            weight_for_0 = (1 / neg) * total / 2.0
            weight_for_1 = (1 / pos) * total / 2.0
            self.class_weights = {0: weight_for_0, 1: weight_for_1}
            print('Using class weights:', self.class_weights)
        else:
            self.class_weights = None
        # variables used for performing influence updates
        self.z_x, self.z_y = None, None  # values of "old" training data
        self.z_x_delta, self.z_y_delta = None, None  # value of changed training data

    # the network used by grosse et. al in the paper 'adversarial examples for malware detection'
    def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
        n_hidden = 200
        model = Sequential()
        model.add(Dense(units=n_hidden, activation='relu', input_shape=(self.dim,), kernel_regularizer=L2(self.lambda_)))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=n_hidden, activation='relu', kernel_regularizer=L2(self.lambda_)))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=2, activation='softmax', kernel_regularizer=L2(self.lambda_)))
        if weight_path is not None:
            model.load_weights(weight_path)
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(class_id=1),
                   tf.keras.metrics.Recall(class_id=1)]
        if optimizer == 'Adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate, epsilon=0.1),
                          loss=CategoricalCrossentropy(), metrics=metrics)
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate), loss=CategoricalCrossentropy(), metrics=metrics)
        return model

    # converts a list of matrices (parameters) to a single long vector containing all parameters.
    @staticmethod
    def list_to_vec(param_list):
        l_reshaped = [tf.reshape(l, (-1,)) for l in param_list]  # noqa:E741
        concatenated = tf.concat(l_reshaped, axis=0)
        return concatenated

    def vec_to_list(self, vec, trainable_only=False):
        model_weights = self.model.trainable_weights if trainable_only else self.model.get_weights()
        weight_shapes = [w.shape for w in model_weights]
        total_n_weights = sum([np.product(s) for s in weight_shapes])
        weight_list = []
        params_processed = 0
        for w in model_weights:
            n_params = np.product(w.shape)
            # params_reshaped = vec[params_processed:params_processed+n_params].reshape(w.shape)
            params_reshaped = tf.reshape(vec[params_processed:params_processed + n_params], w.shape)
            params_processed += n_params
            weight_list.append(params_reshaped)
        assert params_processed == total_n_weights
        return weight_list

    # given list of arrays (parameters of model) concats them into one long vector, normalizes it and converts it
    # back to the correct shapes. Needed for power method to compute eigenvalues
    def norm_param_list(self, param_list):
        long_vec = self.list_to_vec(param_list)
        norm = tf.norm(long_vec)
        param_list_normed = [1./norm*p for p in param_list]
        return param_list_normed

    # calculate largest eigenvalue of hessian using lanczos algorithm using a linear operator (hessian vector product)
    # to approximate the hessian matrix

    def lanczos_largest_eigval(self, batch_size=128):
        v0 = self.norm_param_list([tf.random.uniform(w.shape) for w in self.model.trainable_weights])
        #  v0 = [tf.random.uniform(w.shape) for w in self.model.trainable_weights]
        n_trainable_weights = self.list_to_vec(v0).shape[0]
        hvp_op = HessianVectorProduct(self, n_trainable_weights, batch_size=batch_size)
        # hvp = lambda v: self.hvp_train(v, batch_size=batch_size)
        # hvp_op = LinearOperator(shape=(n_trainable_weights, n_trainable_weights), matvec=hvp)
        # LM = largest magnitude eigenvalue
        eigvals_large = eigsh(hvp_op, k=1, which='LM', return_eigenvectors=False)
        return eigvals_large[0]

    # returns (absolute) largest eigenvalue of hessian via power iteration method
    # mu is a factor that can be substracted, i.e. H -> H - mu*Id. If mu is the largest eigenvalue this computes
    # the smallest eigenvalue - mu. I.e. can be used to compute largest and smallest eigenvalue
    def power_iteration_hessian(self, iterations=100, batch_size=100, mu=0):
        weights = self.model.trainable_weights
        random_init_vector = [tf.random.uniform(w.shape) for w in weights]
        normed_init_vector = self.norm_param_list(random_init_vector)
        value = 42
        i = tf.constant(0)

        def cond(i, eigenvec, eigenvalue): return tf.less(i, iterations)

        def body(i, eigenvec, eigenvalue):
            vector_update = self.hvp_train(eigenvec, batch_size=batch_size)
            if mu != 0:
                vector_update = [vu - mu*vu for vu in vector_update]
            vector_normed = self.norm_param_list(vector_update)
            rayleigh_nominator = tf.reduce_sum(tf.multiply(self.list_to_vec(eigenvec), self.list_to_vec(vector_update)))
            rayleigh_denomniator = tf.reduce_sum(tf.multiply(self.list_to_vec(eigenvec), self.list_to_vec(eigenvec)))
            value_update = rayleigh_nominator / rayleigh_denomniator
            value_update += mu
            tf.print('Eigenvalue after {} epochs: {}'.format(i, value_update))
            return i+1, vector_normed, value_update

        print('Starting power iteration for Hessian with {} iterations.'.format(iterations))
        loop_vars = (i, normed_init_vector, value)
        res = tf.while_loop(cond, body, loop_vars)
        return res[1], res[2]

    # computes largest and smallest eigenvalue of hessian via power iteration method. This is helpful to compute
    # hyperparameters `scale` and `damping_value`. Scale must be chosen such that eigenvalues lie in [-1,1] (because
    # LISSA only converges for this case) and damping must be chosen to be smallest (scaled) eigenvalue
    # (or a bit higher) such that all eigenvalues are positive after adding damping * Id to Hessian.
    def get_hessian_largest_smallest_eigenvalue(self, power_iterations=100):
        _, largest_eigenvalue = self.power_iteration_hessian(iterations=power_iterations)
        print('Largest eigenvalue after {} iterations: {}'.format(power_iterations, largest_eigenvalue))
        _, smallest_eigenvalue = self.power_iteration_hessian(iterations=power_iterations, mu=largest_eigenvalue)
        print('Smallest eigenvalue after {} iterations: {}'.format(power_iterations, smallest_eigenvalue))
        return largest_eigenvalue, smallest_eigenvalue

    # returns copy of train/test/valid data in which certain columns can be set to zero
    def get_data_copy(self, data_name, indices_to_delete, **kwargs):
        assert data_name in ['train', 'test', 'valid']
        if data_name == 'train':
            data_cpy = self.x_train.copy()
        elif data_name == 'test':
            data_cpy = self.x_test.copy()
        else:
            data_cpy = self.x_valid.copy()
        # delete indices if given.
        if len(indices_to_delete) > 0:
            data_cpy[:, indices_to_delete] = 0
        return data_cpy

    # returns copy of train/test/valid data in which certain columns can be set to zero or a value of choice
    def get_data_copy_y(self, data_name, indices_to_delete, new_labels=None):
        assert data_name in ['train', 'test', 'valid']
        if data_name == 'train':
            data_cpy = self.y_train.copy()
        elif data_name == 'test':
            data_cpy = self.y_test.copy()
        else:
            data_cpy = self.y_valid.copy()
        # binary / LR case. -1 becomes 1 and 1 becomes -1 else new_labels
        if len(data_cpy.shape) == 1:
            if new_labels is None:
                data_cpy[indices_to_delete] *= -1
            else:
                data_cpy[indices_to_delete] = new_labels
        # categorical case
        else:
            n_classes = data_cpy.shape[1]
            if new_labels is None:
                if n_classes == 2:
                    old_labels = np.argmax(data_cpy[indices_to_delete], axis=1)
                    random_classes = 1-old_labels  # flip label
                else:
                    random_classes = np.random.choice(range(n_classes), len(indices_to_delete))
                new_labels_categorical = to_categorical(random_classes, n_classes)
            else:
                new_labels_categorical = to_categorical(new_labels, n_classes)
            data_cpy[indices_to_delete] = new_labels_categorical
        return data_cpy

    # returns indices of samples in training_data where the features corresponding to 'indices_to_delete' appear
    def get_relevant_indices(self, indices_to_delete):
        relevant_indices = self.x_train[:, indices_to_delete].nonzero()[0]
        # to avoid having samples more than once
        relevant_indices = np.unique(relevant_indices)
        return relevant_indices

    # compute loss and f1/precision/recall for x and y given parameters theta
    def get_performance(self, x, y, theta):
        model = self.get_network()
        assert len(model.get_weights()) == len(theta)
        model.set_weights(theta)
        if sp.issparse(x):
            datagen = DrebinDataGenerator(x, y, batch_size=100, shuffle=False)
            evaluation = model.evaluate(datagen, verbose=0)
            y_hat = tf.argmax(model.predict(datagen), axis=1)
        else:
            evaluation = model.evaluate(x, y, batch_size=5000, verbose=0)
            y_hat = tf.argmax(model.predict(x, batch_size=5000, verbose=0), axis=1)
        y_class = tf.argmax(y, axis=1)
        report = classification_report(y_class.numpy(), y_hat.numpy(), output_dict=True)
        report['test_loss'] = evaluation[0] if type(evaluation) is list else evaluation
        return report

    # training of the model on the full train dataset
    def train_model(self, model_folder, **kwargs):
        batch_size = 64 if 'batch_size' not in kwargs else kwargs['batch_size']
        epochs = 150 if 'epochs' not in kwargs else kwargs['epochs']
        data_augmentation = False if 'data_augmentation' not in kwargs else kwargs['data_augmentation']

        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        if sp.issparse(self.x_train):
            train_datagen = DrebinDataGenerator(self.x_train, self.y_train,
                                                batch_size, class_weights=self.class_weights)
            test_datagen = DrebinDataGenerator(self.x_test, self.y_test, batch_size=104, shuffle=False)
            best_model, test_loss = self.train_retrain(
                self.model, train_datagen, test_datagen, model_folder, epochs=epochs)
        else:
            best_model, test_loss = self.train_retrain(self.model, (self.x_train, self.y_train),
                                                       (self.x_test, self.y_test), model_folder,
                                                       epochs=epochs, data_augmentation=data_augmentation)
        self.model.set_weights(best_model.get_weights())
        return test_loss

    # retraining without 'indices_to_delete'
    def retrain_model(self, indices_to_delete, save_folder, retrain_labels=False, **kwargs):
        # if called from `unlearn.py` this will be always set
        batch_size = kwargs['batch_size']
        epochs = kwargs['epochs']
        optimizer = kwargs['optimizer']
        affected_samples = kwargs['affected_samples']
        print('Using {} optimizer for retraining.'.format(optimizer))
        if retrain_labels:
            assert np.min(indices_to_delete) >= 0 and np.max(indices_to_delete) <= self.n - 1
        # else:
        #    assert np.min(indices_to_delete) >= 0 and np.max(indices_to_delete) <= self.dim-1
        no_features_to_delete = len(indices_to_delete)
        combination_string = hashlib.sha256('-'.join([str(i) for i in indices_to_delete]).encode()).hexdigest()
        if affected_samples is not None:
            l = len(affected_samples)  # noqa:E741
            subfolder_name = str(no_features_to_delete) + '_' + str(l)
        else:
            subfolder_name = str(no_features_to_delete)
        model_folder = os.path.join(save_folder, subfolder_name, combination_string)
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        # if retrain_labels is True the indices_to_delete refer to labels to flip
        if retrain_labels:
            x_train_delta = self.x_train
            x_test_delta = self.x_test
            y_train_delta = self.get_data_copy_y('train', indices_to_delete)
            y_test_delta = self.y_test
        else:
            x_train_delta = self.get_data_copy('train', indices_to_delete, affected_samples=affected_samples)
            x_test_delta = self.get_data_copy('test', indices_to_delete)
            y_train_delta = self.y_train
            y_test_delta = self.y_test
        # get copy of original model to start training from
        tmp_model = self.get_network(optimizer=optimizer)
        tmp_model.set_weights(self.model.get_weights())
        if sp.issparse(self.x_train):
            train_datagen = DrebinDataGenerator(x_train_delta, y_train_delta,
                                                batch_size, class_weights=self.class_weights)
            test_datagen = DrebinDataGenerator(x_test_delta, y_test_delta, batch_size=104, shuffle=False)
            best_model, new_loss = self.train_retrain(
                tmp_model, train_datagen, test_datagen, model_folder, epochs=epochs)
        else:
            best_model, new_loss = self.train_retrain(tmp_model, (x_train_delta, y_train_delta),
                                                      (x_test_delta, y_test_delta), model_folder, epochs=epochs,
                                                      batch_size=batch_size)
        return best_model, new_loss

    # basic training routines
    def train_retrain(self, model, train, test, model_folder, epochs=150, batch_size=64, data_augmentation=False):
        model_save_path = os.path.join(model_folder, 'best_model.hdf5')
        csv_save_path = os.path.join(model_folder, 'train_log.csv')
        json_report_path = os.path.join(model_folder, 'test_performance.json')
        metric_for_min = 'loss'
        # metric_for_min = 'val_loss'
        model_checkpoint_loss = ModelCheckpoint(model_save_path, monitor=metric_for_min, save_best_only=True,
                                                save_weights_only=True)
        csv_logger = CSVLogger(csv_save_path)
        callbacks = [model_checkpoint_loss, csv_logger]
        # if x is sparse train test is a generator
        if sp.issparse(self.x_train):
            hist = model.fit(train, epochs=epochs, validation_data=test, verbose=1, callbacks=callbacks).history
        # else its a numpy array
        elif data_augmentation:
            # augmentation only implemented for this mode
            datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)
            datagen.fit(train[0])
            lr_scheduler = LearningRateScheduler(lr_schedule)
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                           cooldown=0,
                                           patience=5,
                                           min_lr=0.5e-6)
            callbacks = [lr_scheduler, lr_reducer] + callbacks
            hist = model.fit(datagen.flow(train[0], train[1], batch_size=batch_size),
                             epochs=epochs, validation_data=(test[0], test[1]), verbose=1,
                             callbacks=callbacks,
                             steps_per_epoch=len(train[0]) / batch_size).history
        else:
            hist = model.fit(train[0], train[1], epochs=epochs, validation_data=(test[0], test[1]), verbose=1,
                             callbacks=callbacks, batch_size=batch_size).history
        best_loss = np.min(hist[metric_for_min]) if metric_for_min in hist else np.inf
        best_loss_epoch = np.argmin(hist[metric_for_min]) + 1 if metric_for_min in hist else 0
        print('Best model has test loss {} after {} epochs'.format(best_loss, best_loss_epoch))
        best_model = self.get_network()
        best_model.load_weights(model_save_path)
        # in sparse case x test and train are generators
        if sp.issparse(self.x_train):
            y_test_hat = np.argmax(best_model.predict(test), axis=1)
            test_loss = best_model.evaluate(test, batch_size=1000, verbose=0)[0]
        else:
            y_test_hat = np.argmax(best_model.predict(test[0]), axis=1)
            test_loss = best_model.evaluate(test[0], test[1], batch_size=1000, verbose=0)[0]
        report = classification_report(np.argmax(self.y_test, axis=1), y_test_hat, digits=4, output_dict=True)
        report['train_loss'] = best_loss
        report['test_loss'] = test_loss
        report['epochs_for_min'] = int(best_loss_epoch)  # json does not like numpy ints
        json.dump(report, open(json_report_path, 'w'), indent=4)
        return best_model, best_loss

    # performs of SGD on (x_train, y_train)
    def fine_tune(self, x_train, y_train, learning_rate, batch_size=256, epochs=1):
        tmp_model = self.get_network(optimizer='SGD', learning_rate=learning_rate)
        tmp_model.set_weights(self.model.get_weights())
        tmp_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        weights = tmp_model.get_weights()
        return weights

    # returns dL/dW (x, y)
    def get_gradients(self, x_tensor, y_tensor, batch_size=2048):
        grads = []
        for start in range(0, x_tensor.shape[0], batch_size):
            with LoggedGradientTape() as tape:
                tape.watch(self.model.trainable_weights)
                result = self.model(x_tensor[start:start+batch_size])
                loss = self.model.loss(y_tensor[start:start+batch_size], result)
                grads.append(tape.gradient(loss, self.model.trainable_weights))
        grads = list(zip(*grads))
        for i in range(len(grads)):
            grads[i] = tf.add_n(grads[i])
        # for embedding layers, gradient can be of type indexed slices and need to be converted
        for i in range(len(grads)):
            if type(grads[i]) == tf.IndexedSlices:
                grads[i] = tf.convert_to_tensor(grads[i])
        return grads

    # computes d/dW [ Loss(x_delta, y_delta) - Loss(x,y) ]
    # this saves one gradient call compared to calling `get_gradients` twice
    def get_gradients_diff(self, x_tensor, y_tensor, x_delta_tensor, y_delta_tensor, batch_size=2048):
        assert x_tensor.shape == x_delta_tensor.shape and y_tensor.shape == y_delta_tensor.shape
        grads = []
        for start in range(0, x_tensor.shape[0], batch_size):
            with LoggedGradientTape() as tape:
                tape.watch(self.model.trainable_weights)
                result_x = self.model(x_tensor[start:start + batch_size])
                result_x_delta = self.model(x_delta_tensor[start:start + batch_size])
                loss_x = self.model.loss(y_tensor[start:start + batch_size], result_x)
                loss_x_delta = self.model.loss(y_delta_tensor[start:start + batch_size], result_x_delta)
                diff = loss_x_delta - loss_x
                grads.append(tape.gradient(diff, self.model.trainable_weights))
        grads = list(zip(*grads))
        for i in range(len(grads)):
            grads[i] = tf.add_n(grads[i])
        # for embedding layers, gradient can be of type indexed slices and need to be converted
        for i in range(len(grads)):
            if type(grads[i]) == tf.IndexedSlices:
                grads[i] = tf.convert_to_tensor(grads[i])
        return grads


    # hessian vector product between H and v for arrays x and y
    def hvp(self, x, y, v):
        # 1st gradient of Loss w.r.t weights
        with LoggedGradientTape() as tape:
            # first gradient
            grad_L = self.get_gradients(x, y)
            assert len(v) == len(grad_L)
            # v^T * \nabla L
            v_dot_L = [v_i * grad_i for v_i, grad_i in zip(v, grad_L)]
            # tape.watch(self.model.weights)
            # second gradient computation
            hvp = tape.gradient(v_dot_L, self.model.trainable_weights)
        # for embedding layers, gradient can be of type indexed slices and need to be converted
        for i in range(len(hvp)):
            if type(hvp[i]) == tf.IndexedSlices:
                hvp[i] = tf.convert_to_tensor(hvp[i])
        return hvp

    # computes H*v where H is computed as the sum over the entire training dataset. This means we have to calculate
    # sum_{x,y \in train}(H(x,y)*v)
    def hvp_train(self, v, batch_size):
        n_batches = np.ceil(self.n // batch_size)

        def cond(i, sum):
            return tf.less(i, n_batches)

        def body(i, sum):
            start, end = i * batch_size, (i + 1) * batch_size
            if sp.issparse(self.x_train):
                batch_hvps = self.hvp(self.x_train[start:end].toarray(),
                                      self.y_train[start:end], v)
            else:
                batch_hvps = self.hvp(self.x_train[start:end],
                                      self.y_train[start:end], v)
            new_sum = [a + b for (a, b) in zip(sum, batch_hvps)]
            return i + 1, new_sum

        i = tf.constant(0)
        init_sum = [tf.zeros_like(p) for p in v]
        loop_vars = (i, init_sum)
        res = tf.while_loop(cond, body, loop_vars)
        return res[1]

    # calculates H^-1*v using the iterative scheme proposed by Agarwal et al with batch updates
    # the scale and damping parameters have to be found by trial and error to achieve convergence
    # rounds can be set to average the results over multiple runs to decrease variance and stabalize the results
    def get_inv_hvp_lissa(self, x, y, v, batch_size, scale, damping, iterations=-1, verbose=False,
                          rounds=1, early_stopping=True, patience=5):
        i = tf.constant(0)
        n_batches = np.ceil(x.shape[0] / batch_size) if iterations == -1 else iterations
        shuffle_indices = [np.random.permutation(range(x.shape[0])) for _ in range(rounds)]
        def cond(i, u, shuff_idx, update_min): return tf.less(i, n_batches) and tf.math.is_finite(tf.norm(u[0]))

        def body(i, u, shuff_idx, update_min):
            start, end = i * batch_size, (i+1) * batch_size
            if sp.issparse(x):
                batch_hvps = self.hvp(x[shuff_idx][start:end].toarray(),
                                      y[shuff_idx][start:end], u)
            else:
                batch_hvps = self.hvp(x[shuff_idx][start:end],
                                      y[shuff_idx][start:end], u)
            new_estimate = [a + (1-damping) * b - c/scale for (a, b, c) in zip(v, u, batch_hvps)]
            update_norm = np.sum(np.sum(np.abs(old - new)) for old, new in zip(u, new_estimate))
            if early_stopping and update_norm > update_min[0] and update_min[-1] >= patience:
                tf.print(f"Early stopping at iteration {i+1}. Update norm {update_norm} > {update_min}")
                if i < patience:
                    i = n_batches + 1
                else:
                    i = n_batches
            if update_norm < update_min[0]:
                update_min = [update_norm, 1]
            if verbose:
                tf.print(i, [tf.norm(ne) for ne in new_estimate][:5])
            update_min[-1] += 1
            return i+1, new_estimate, shuff_idx, update_min

        estimate = None
        for r in range(rounds):
            loop_vars = (i, v, shuffle_indices[r], [np.inf, -1])
            res = tf.while_loop(cond, body, loop_vars)
            # i encodes the exit reason of the body:
            #   i == n_batches:     maximum number of iterations reached
            #   i == n_batches+1:   early stopping criterium reached
            #   i == n_batches+2:   early stopping after first iterations (diverged)
            if res[0] == n_batches+2:
                return res[1], True
            # if one iteration failed averaging makes no sense anymore
            if not all([tf.math.is_finite(tf.norm(e)) for e in res[1]]):
                return res[1], True
            res_upscaled = [r/scale for r in res[1]]
            if estimate is None:
                estimate = [r/rounds for r in res_upscaled]
            else:
                for j in range(len(estimate)):
                    estimate[j] += res_upscaled[j] / rounds
        diverged = not all([tf.math.is_finite(tf.norm(e)) for e in estimate])
        return estimate, diverged

    # scipy.optimize.fmin_ncg requires vectors as inputs. Therefore we have to reshape the parameters into one vector
    def get_fmin_loss_fn(self, v, damping):

        def get_fmin_loss(t):
            t_list = self.vec_to_list(t)
            if sp.issparse(self.x_train):
                hvp = self.hvp(self.x_train.toarray(), self.y_train, t_list)
            else:
                hvp = self.hvp(self.x_train, self.y_train, t_list)
            hvp_damp = [h + damping * v_ for h, v_ in zip(hvp, v)]
            res = 0.5 * np.dot(self.list_to_vec(hvp_damp), t) - np.dot(self.list_to_vec(v), t)
            return res

        return get_fmin_loss

    # the gradient of a quadratic form 1/2*x^TAx + b^Tx is 1/2*A^Tx + 1/2*Ax +b. If A symmetric: Ax+b
    def get_fmin_loss_grad(self, v, damping):

        def get_fmin_grad(t):
            t_list = self.vec_to_list(t)
            if sp.issparse(self.x_train):
                hvp = self.hvp(self.x_train.toarray(), self.y_train, t_list)
            else:
                hvp = self.hvp(self.x_train, self.y_train, t_list)
            hvp_damp = [h + damping * v_ for h, v_ in zip(hvp, v)]
            res = 0.5 * self.list_to_vec(hvp_damp) - np.dot(self.list_to_vec(v), t)
            return res

        return get_fmin_grad

    def get_fmin_hvp_(self, v, damping):

        def get_fmin_hvp(x, t):
            t_list = self.vec_to_list(t)
            if sp.issparse(self.x_train):
                hvp = self.hvp(self.x_train.toarray(), self.y_train, t_list)
            else:
                hvp = self.hvp(self.x_train, self.y_train, t_list)
            hvp_damp = [h + damping * v_ for h, v_ in zip(hvp, v)]
            return self.list_to_vec(hvp_damp)
        return get_fmin_hvp

    def get_inv_hvp_cg(self, v, damping):
        # we have to provide minimization objective, gradient, and hessian*vec for scipy method
        fmin_loss_fn = self.get_fmin_loss_fn(v, damping)
        fmin_grad_fn = self.get_fmin_loss_grad(v, damping)
        fmin_hvp = self.get_fmin_hvp_(v, damping)
        fmin_approx = fmin_ncg(
            f=fmin_loss_fn,
            x0=self.list_to_vec(v),
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp,
            avextol=1e-8,
            maxiter=100,
            retall=True
        )

        return self.vec_to_list(fmin_approx)

    # performs parameter update using influence functions. Required self.z_x, self.z_y, self.z_x_delta, self.z_y_delta
    # to be set correctly, i.e. call update_influence_variables in some fashion beforehand
    def approx_retraining(self, **kwargs):
        batch_size = 500 if 'batch_size' not in kwargs else kwargs['batch_size']
        scale = 10 if 'scale' not in kwargs else kwargs['scale']
        damping = 0.1 if 'damping' not in kwargs else kwargs['damping']
        iterations = -1 if 'iterations' not in kwargs else kwargs['iterations']
        verbose = False if 'verbose' not in kwargs else kwargs['verbose']
        rounds = 1 if 'rounds' not in kwargs else kwargs['rounds']
        conjugate_gradients = False if 'cg' not in kwargs else kwargs['cg']
        order = 2 if 'order' not in kwargs else kwargs['order']
        tau = 1 if 'tau' not in kwargs else kwargs['tau']  # unlearning rate
        hvp_x = self.x_train if 'hvp_x' not in kwargs else kwargs['hvp_x']
        hvp_y = self.y_train if 'hvp_y' not in kwargs else kwargs['hvp_y']

        if order == 1:
            # first order update
            diff = self.get_gradients_diff(self.z_x, self.z_y, self.z_x_delta, self.z_y_delta)
            d_theta = diff
            diverged = False
        elif order == 2:
            # second order update
            diff = self.get_gradients_diff(self.z_x, self.z_y, self.z_x_delta, self.z_y_delta)
            # skip hvp if diff == 0
            if np.sum(np.sum(d) for d in diff) == 0:
                d_theta = diff
                diverged = False
            elif conjugate_gradients:
                d_theta = self.get_inv_hvp_cg(diff, damping)
                diverged = True
            else:
                d_theta, diverged = self.get_inv_hvp_lissa(
                    hvp_x, hvp_y, diff, batch_size, scale, damping, iterations, verbose, rounds)
        if order != 0:
            # only update trainable weights (non-invasive workaround for BatchNorm layers in CIFAR model)
            d_theta = [d_theta.pop(0) if w.trainable else 0 for w in self.model.weights]
            theta_approx = [w - tau * d_t for w, d_t in zip(self.model.get_weights(), d_theta)]
        return theta_approx, diverged

    def iter_approx_retraining(self, z_x, z_y, z_x_delta, z_y_delta, delta_idx, prioritize_misclassified=True,
                               steps=1, mixing_ratio=1.0, cm_dir=None, verbose=False, **unlearn_kwargs):
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

        if cm_dir is not None:
            self.plot_cm(plot_validation=True, title='Before Unlearning',
                         outfile=os.path.join(cm_dir, 'cm_unlearning_00.png'))

        new_theta = self.model.get_weights().copy()
        # the TmpState context managers restore the states of weights, z_x, z_y, ... afterwards
        with ModelTmpState(self), DeltaTmpState(self):
            for step in range(steps):
                prio_idx = None
                Y_pred = batch_pred(self.model, z_x).numpy()
                if prioritize_misclassified:
                    n_to_add = int(mixing_ratio * delta_idx.shape[0])
                    preds = np.argmax(self.model.predict(z_x, batch_size=500), axis=1)
                    prio_idx_all = np.argwhere(preds != np.argmax(z_y_delta, axis=1))[:, 0]
                    prio_wo_delta = list(set(prio_idx_all) - set(delta_idx))
                    prio_idx = np.random.choice(prio_wo_delta, n_to_add)
                    print(f'Adding {n_to_add} wrong samples to {delta_idx.shape[0]} delta indices')
                idx = np.hstack([prio_idx, delta_idx]) if prio_idx is not None else delta_idx
                self.z_x = z_x[idx]
                self.z_x_delta = z_x_delta[idx]
                """
                # alternative to using model predictions we could use the training labels:
                self.z_y = z_y[idx]
                """
                # for the wrong predictions we use the models current predictions as z_y
                z_y_1 = to_categorical(np.argmax(self.model(z_x[prio_idx]), axis=1), z_y_delta.shape[-1])
                # for the canary samples we stick to the old canary samples
                z_y_2 = z_y[delta_idx]
                self.z_y = np.vstack([z_y_1, z_y_2])
                self.z_y_delta = z_y_delta[idx]
                new_theta, diverged = self.approx_retraining(hvp_x=z_x_delta, hvp_y=z_y_delta, **unlearn_kwargs)
                self.model.set_weights(new_theta)
                if verbose or True:
                    preds = np.argmax(self.model.predict(self.z_x, verbose=0), axis=1)
                    correct_preds = np.where(preds == np.argmax(self.z_y_delta, axis=1))[0]
                    acc = len(correct_preds) / self.z_x.shape[0]
                    print(f">> iterative approx retraining: step = {step+1}, accuracy = {acc}")
                if cm_dir is not None:
                    self.plot_cm(plot_validation=True, title=f'After Unlearning Step {step+1}',
                                 outfile=os.path.join(cm_dir, f'cm_unlearning_{step+1:02d}.png'))
        return new_theta, diverged

    def get_mixed_delta_idx(self, delta_idx, n_samples, mixing_ratio=1.0, prio_idx=None):
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
        remaining_idx = list(set(range(n_samples)) - set(delta_idx) - set(priority_idx))
        n_total = np.ceil(mixing_ratio*delta_idx.shape[0]).astype(np.int) + delta_idx.shape[0]
        n_prio = min(n_total, len(priority_idx))
        n_regular = max(n_total - len(priority_idx) - len(remaining_idx), 0)
        idx = np.hstack((
            delta_idx,
            np.random.choice(priority_idx, n_prio, replace=False),
            #np.random.choice(remaining_idx, n_regular, replace=False)
        ))
        return idx.astype(np.int)

    # updates self.z_x, self.z_y, self.z_x_delta, self.z_y_delta by changing samples of training data directly
    # @param indices_to_change: The rows in x_train connected to these indices will be changed
    # @param new_x: The rows in x_train connected to indices_to_change indices will be changed to this array
    # @param new_y: The rows in y_train connected to indices_to_change indices will be changed to this array
    def update_influence_variables_samples_indices(self, indices_to_change, new_x, new_y=None):
        assert np.min(indices_to_change) >= 0 and np.max(indices_to_change) < self.n
        assert self.x_train[indices_to_change].shape == new_x.shape, '{} != {}'.format(
            self.x_train[indices_to_change].shape,
            new_x.shape)
        self.z_x, self.z_y = self.x_train[indices_to_change], self.y_train[indices_to_change]
        self.z_x_delta = new_x
        if new_y is not None:
            assert self.y_train[indices_to_change].shape == new_y.shape
            self.z_y_delta = new_y
        else:
            self.z_y_delta = self.y_train[indices_to_change]
        if sp.issparse(self.x_train):
            self.z_x, self.z_x_delta = self.z_x.toarray(), self.z_x_delta.toarray()

    # updates self.z_x, self.z_y, self.z_x_delta, self.z_y_delta directly to given arguments
    # @param z_x: Value for z_x
    # @param z_y: Value for z_y
    # @param new_x: Value for z_x_delta
    # @param new_y: Value for z_y_delta
    def update_influence_variables_samples(self, z_x, z_y, z_x_delta, z_y_delta=None):
        self.z_x = z_x
        self.z_y = z_y
        self.z_x_delta = z_x_delta
        self.z_y_delta = z_y_delta if z_y_delta is not None else z_y
        if sp.issparse(self.x_train):
            self.z_x, self.z_x_delta = self.z_x.toarray(), self.z_x_delta.toarray()

    # updates self.z_x, self.z_y, self.z_x_delta, self.z_y_delta by an array of dimensions to be deleted, i.e. set to 0
    # @param dimensions_to_delete: The columns in x_train connected to these dimensions will be deleted
    def update_influence_variables_dimension_deletion(self, dimensions_to_delete, affected_samples=None):
        # assert np.min(dimensions_to_delete) >= 0 and np.max(dimensions_to_delete) < self.dim
        # if affected samples is given, set dimensions only in those to zero
        if affected_samples is not None:
            new_x = self.get_data_copy('train', dimensions_to_delete)[affected_samples]
            self.update_influence_variables_samples_indices(affected_samples, new_x)
        else:
            relevant_indices = self.get_relevant_indices(dimensions_to_delete)
            new_x = self.get_data_copy('train', dimensions_to_delete)[relevant_indices]
            self.update_influence_variables_samples_indices(relevant_indices, new_x)

    # updates self.z_x, self.z_y, self.z_x_delta, self.z_y_delta by an array of sample indices for which the label
    # will be flipped
    # @param dimensions_to_delete: The labels in y_train connected to these indices will be flipped
    def update_influence_variables_label_flip(self, indices_to_flip, new_labels=None):
        assert np.min(indices_to_flip) >= 0 and np.max(indices_to_flip) < self.n
        new_y = self.get_data_copy_y('train', indices_to_flip, new_labels=new_labels)[indices_to_flip]
        new_x = self.x_train[indices_to_flip]  # z_x_delta does not change
        self.update_influence_variables_samples_indices(indices_to_flip, new_x, new_y)


class HessianVectorProduct(LinearOperator):
    def __init__(self, unlearner, n_trainable_weights, batch_size=128, dtype='float32'):
        self.unlearner = unlearner
        self.batch_size = batch_size
        self.dtype = np.dtype(dtype)
        self.shape = (n_trainable_weights, n_trainable_weights)

    def _matvec(self, v):
        # reshape v into weight shapes, compute hvp and reshape back
        v = self.unlearner.vec_to_list(v, trainable_only=True)
        hvp = self.unlearner.hvp_train(v, batch_size=self.batch_size)
        hvp = self.unlearner.list_to_vec(hvp)
        return hvp

    def _rmatvec(self, v):
        return self._matvec(v)


def batch_pred(model, x, batch_size=2048):
    preds = []
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        preds.append(model(x[start:end]))
    return tf.concat(preds, 0)
