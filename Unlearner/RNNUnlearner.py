import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import json
import sys
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam, SGD
from Unlearner.DNNUnlearner import DNNUnlearner
from Unlearner.CanaryCallback import CanaryCallback
from sklearn.metrics import classification_report
from scipy.stats import skewnorm


class RNNUNlearner(DNNUnlearner):
    def __init__(self, x_train, y_train, embedding_dim, idx2char, lambda_=0.01, weight_path=None, canary_start=None,
                 canary_number=None, canary_repetitions=None, n_layers=1, n_units=256, p_dropout=0.0):
        tf.random.set_seed(42)
        # training data
        self.x_train = x_train  # all x data is of shape (n_samples, max_len) and stored as unique indices
        self.y_train = y_train
        # test data makes no real sense in this setting so we use training data
        self.x_test = x_train.copy()
        self.y_test = y_train.copy()
        self.idx2char = idx2char
        self.char2idx = {v:k for k,v in self.idx2char.items()}
        self.n = self.x_train.shape[0]
        # model params
        self.max_len = self.x_train.shape[1]
        self.dim = len(idx2char)  # here dim refers to the number of words in the vocabulary
        self.embedding_dim = embedding_dim
        self.lambda_ = lambda_
        self.n_units = n_units
        self.n_layers = n_layers
        self.model = self.get_network(weight_path=weight_path, no_lstm_units=n_units, n_layers=n_layers, p_dropout=p_dropout)
        # canary stuff
        self.canary_start = canary_start
        self.canary_number = canary_number
        self.canary_repetitions = canary_repetitions
        self.param_string = 'lambda={}-canary_number={}-canary_reps={}-embedding_dim={}-seqlen={}-dropout={}'.format(
            lambda_, canary_number, canary_repetitions, embedding_dim, x_train.shape[1], p_dropout)

    def get_network(self, weight_path=None, optimizer='Adam', no_lstm_units=512, n_layers=2, p_dropout=0.0,
                    learning_rate=0.0001):
        # define the LSTM model
        model = Sequential()
        model.add(Embedding(input_dim=self.dim, output_dim=self.embedding_dim))
        if n_layers > 1:
            model.add(LSTM(no_lstm_units, kernel_regularizer=L2(self.lambda_), recurrent_regularizer=L2(self.lambda_),
                           bias_regularizer=L2(self.lambda_), return_sequences=True))
        else:
            model.add(LSTM(no_lstm_units, kernel_regularizer=L2(self.lambda_), recurrent_regularizer=L2(self.lambda_),
                       bias_regularizer=L2(self.lambda_)))
        for _ in range(n_layers - 1):
            model.add(LSTM(no_lstm_units, kernel_regularizer=L2(self.lambda_), recurrent_regularizer=L2(self.lambda_),
                                bias_regularizer=L2(self.lambda_)))
        if p_dropout > 0:
            model.add(Dropout(p_dropout))
        model.add(Dense(self.dim, activation='softmax', kernel_regularizer=L2(self.lambda_), bias_regularizer=L2(self.lambda_)))
        if weight_path is not None:
            # load the network weights
            if weight_path.endswith('ckpt'):
                model.load_weights(weight_path).expect_partial()
            elif weight_path.endswith('hdf5'):
                model.load_weights(weight_path)
            else:
                print('Invalid file format')
        metrics = [tf.keras.metrics.CategoricalAccuracy()]
        if optimizer == 'Adam':
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=metrics)
        else:
            model.compile(loss=categorical_crossentropy, optimizer=SGD(learning_rate=learning_rate), metrics=metrics)
        return model

    # training of the model on the full train dataset. Overwrites parent method since no test dataset exists
    def train_model(self, model_folder, **kwargs):
        batch_size = 64 if 'batch_size' not in kwargs else kwargs['batch_size']
        epochs = 150 if 'epochs' not in kwargs else kwargs['epochs']
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        print(self.model.summary())
        best_model, test_loss = self.train_retrain(self.model, (self.x_train, self.y_train), (self.x_train, self.y_train),
                                                   model_folder, batch_size=batch_size, epochs=epochs)
        self.model.set_weights(best_model.get_weights())
        return test_loss

    # train/retrain routine. Add the canary callback for this specific case and no evaluation on test set
    def train_retrain(self, model, train, test, model_folder, epochs=150, batch_size=64):
        checkpoint_name = 'checkpoint_{}.ckpt'.format(self.param_string)
        model_save_path = os.path.join(model_folder, checkpoint_name)
        csv_save_path = os.path.join(model_folder, 'train_log.csv')
        json_report_path = os.path.join(model_folder, 'test_performance.json')
        min_metric = 'loss'
        checkpoint = ModelCheckpoint(model_save_path, monitor=min_metric, save_best_only=True, save_weights_only=True,
                                     mode='min')
        csv_logger = CSVLogger(csv_save_path)
        callbacks_list = [checkpoint, csv_logger]
        if self.canary_start is not None and self.canary_repetitions > 0:
            canary_callback = CanaryCallback(self.canary_start, self.idx2char, self.canary_number)
            callbacks_list.append(canary_callback)
        hist = model.fit(train[0], train[1], epochs=epochs, verbose=1, callbacks=callbacks_list, batch_size=batch_size).history
        best_loss_epoch = np.argmin(hist[min_metric]) + 1  if min_metric in hist else 0# history list starts with 0
        best_train_loss = np.min(hist[min_metric]) if min_metric in hist else np.inf
        best_model = self.get_network(no_lstm_units=self.n_units, n_layers=self.n_layers)
        best_model.load_weights(model_save_path).expect_partial()
        best_test_loss = best_model.evaluate(train[0], train[1], batch_size=1000)
        print('Best model has test loss {} after {} epochs'.format(best_train_loss, best_loss_epoch))
        y_test_hat = np.argmax(best_model.predict(test[0]), axis=1)
        report = classification_report(np.argmax(self.y_test, axis=1), y_test_hat, digits=4, output_dict=True)
        report['train_loss'] = best_train_loss
        report['test_loss'] = best_test_loss
        report['epochs_for_min'] = int(best_loss_epoch)  # json does not like numpy ints
        json.dump(report, open(json_report_path, 'w'), indent=4)
        return best_model, best_test_loss

    # performs of SGD on (x_train, y_train)
    def fine_tune(self, x_train, y_train, learning_rate, batch_size=256, epochs=1):
        tmp_model = self.get_network(optimizer='SGD', no_lstm_units=self.n_units, n_layers=self.n_layers,
                                     learning_rate=learning_rate)
        tmp_model.set_weights(self.model.get_weights())
        tmp_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        weights = tmp_model.get_weights()
        return weights

    def get_relevant_indices(self, indices_to_delete):
        relevant_indices = []
        # its much easier to find occurrences in text
        train_text = [''.join([self.idx2char[i] for i in row.flatten()]) for row in self.x_train]
        for idx, sent in enumerate(train_text):
            for word_to_delete in indices_to_delete:
                if word_to_delete in sent:
                    relevant_indices.append(idx)
                    break
        return relevant_indices

    def get_data_copy(self, data_name, indices_to_delete, **kwargs):
        replacement_char = 'x'
        # its much easier to find occurrences in text
        train_text = [''.join([self.idx2char[i] for i in row.flatten()]) for row in self.x_train]
        total_occurences = 0
        for idx, sent in enumerate(train_text):
            for word_to_delete in indices_to_delete:
                if word_to_delete in sent:
                    word_len = len(word_to_delete)
                    total_occurences += 1
                    train_text[idx] = train_text[idx].replace(word_to_delete, replacement_char*word_len)
        print('The given word(s) occurred {} times ({} % of all samples)'.format(total_occurences, 100*total_occurences/self.n))
        # now back to indices
        data_cpy = np.array([[[self.char2idx[idx]] for idx in s] for s in train_text])
        assert data_cpy.shape == self.x_train.shape
        return data_cpy

    # generates the most likely string given a start_str
    def generate_data(self, start_str=None, weights=None):
        if start_str is None:
            # pick a random seed
            start = np.random.randint(0, self.n)
            pattern = self.x_train[start].squeeze()
        else:
            pattern = np.array([self.char2idx[c] for c in start_str])
        if weights is not None:
            model = self.get_network(no_lstm_units=self.n_units, n_layers=self.n_layers)
            model.set_weights(weights)
        else:
            model = self.model
        print("Seed:")
        print("\"", ''.join([self.idx2char[value] for value in pattern]), "\"")
        # generate characters
        print('Prediction:\n')
        for i in range(50):
            x = np.reshape(pattern, (1, len(pattern), 1))
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = self.idx2char[index]
            sys.stdout.write(result)
            pattern = np.append(pattern, index)
            pattern = pattern[1:len(pattern)]
        print("\nDone.")

    # special purpose method. given start string samples strings that do not incorporate the most likely outcome
    def generate_replacement_string(self, top_k=5, chars_to_create=50):
        pattern = np.array([self.char2idx[s] for s in self.canary_start])
        print("Seed:")
        print("\"", ''.join([self.idx2char[value] for value in pattern]), "\"")
        # generate characters
        print('Prediction:\n')
        for i in range(chars_to_create):
            x = np.reshape(pattern, (1, len(pattern), 1))
            prediction = self.model.predict(x, verbose=0).squeeze()
            indices = np.argsort(-prediction)
            top_indices = indices[1:1+top_k] if i == 0 else indices[:top_k]
            index = np.random.choice(top_indices)
            result = self.idx2char[index]
            sys.stdout.write(result)
            pattern = np.append(pattern, index)
            pattern = pattern[1:len(pattern)]
        print("\nDone.")

    def test_canary(self, reference_char, weights=None, chars_to_predict=40, train_reduction=None):
        if weights is not None:
            model = self.get_network(no_lstm_units=self.n_units, n_layers=self.n_layers)
            model.set_weights(weights)
        else:
            model = self.model
        train_selection = slice(0, train_reduction)  # on CPU it takes very long to classify entire dataset
        train_loss = model.evaluate(self.x_train[train_selection], self.y_train[train_selection], batch_size=1000,
                                    verbose=0)
        train_predictions = np.argmax(model.predict(self.x_train[train_selection], batch_size=1000, verbose=0), axis=1)
        train_labels = np.argmax(self.y_train[train_selection], axis=1)
        train_acc = len(np.where(train_labels == train_predictions)[0]) / self.y_train[train_selection].shape[0]
        n_digits = len(self.canary_number)
        # code copied from CanaryCallback. Seems like there is no way to call it by hand (and get return value)
        ref_char = reference_char if len(reference_char) == 1 else reference_char[0]
        number_char_indices = [self.char2idx[i] for i in [c for c in self.canary_number]]
        ref_char_index = self.char2idx[ref_char]
        start_seq = np.array([self.char2idx[s] for s in self.canary_start])
        start_seq = start_seq.reshape((1, len(start_seq), 1))
        digit_distribution = np.zeros((n_digits, 2))
        argmax_chars = ''
        # generate characters
        for i in range(chars_to_predict):
            index_distribution = model.predict(start_seq, verbose=0)
            char_index = np.argmax(index_distribution)
            if i < digit_distribution.shape[0]:
                # monitor probability of canary char and repcement char
                digit_distribution[i, 0] = index_distribution[0, number_char_indices[i]]
                digit_distribution[i, 1] = index_distribution[0, ref_char_index]
            start_seq = np.append(start_seq, char_index.reshape(1, 1, 1), axis=1)
            start_seq = start_seq[:, 1:start_seq.shape[1] + 1, :]
            argmax_chars += self.idx2char[char_index]
        print('Seed: {}'.format(self.canary_start))
        print('Prediction: {}'.format(argmax_chars))
        print('Train loss: {}'.format(train_loss))
        print('Train acc: {}'.format(train_acc))
        print('Digit probas: {}'.format(digit_distribution[:, 0]))
        print('Replacement_char proba: {}'.format(digit_distribution[:, 1]))
        print('Canary perplexities at all points:')
        for j in range(1,n_digits+1):
            print('{}: {}'.format(j, -np.sum(np.log2(digit_distribution[:j, 0]))))
        canary_perplexity = -np.sum(np.log2(digit_distribution[:, 0]))
        return canary_perplexity, train_loss, train_acc, argmax_chars


    def calc_sequence_perplexity(self, sequence, start_sequence=None):
        # code copied from CanaryCallback. Seems like there is no way to call it by hand (and get return value)
        number_char_indices = [self.char2idx[i] for i in sequence]
        start_seq = np.array([self.char2idx[s] for s in (self.canary_start if start_sequence is None else start_sequence)])
        start_seq = start_seq.reshape((1, len(start_seq), 1))
        digit_distribution = np.zeros(len(sequence))
        argmax_chars = ''
        # generate characters
        for i in range(len(sequence)):
            index_distribution = self.model.predict(start_seq, verbose=0)
            char_index = np.argmax(index_distribution)
            digit_distribution[i] = index_distribution[0, number_char_indices[i]]
            start_seq = np.append(start_seq, char_index.reshape(1, 1, 1), axis=1)
            start_seq = start_seq[:, 1:start_seq.shape[1] + 1, :]
            argmax_chars += self.idx2char[char_index]
        print('Seed: {}'.format(self.canary_start))
        print('Prediction: {}'.format(argmax_chars))
        print('Digit probas: {}'.format(digit_distribution))
        print('Canary perplexities at all points:')
        for j in range(1,len(sequence)+1):
            print('{}: {}'.format(j, -np.sum(np.log2(digit_distribution[:j]))))
        sequence_perplexity = -np.sum(np.log2(digit_distribution))
        return sequence_perplexity


    def calc_perplexity_distribution(self, weights=None, no_samples=1000000, plot=False, only_digits=False):
        if weights is not None:
            model = self.get_network(no_lstm_units=self.n_units, n_layers=self.n_layers)
            model.set_weights(weights)
        else:
            model = self.model
        if only_digits:
            numbers = np.unique([d for d in self.canary_number])
            char_indices = [self.char2idx[n] for n in numbers]
        else:
            char_indices = list(self.idx2char.keys())
        len_canary = len(self.canary_number)
        start_seq = np.array([self.char2idx[s] for s in self.canary_start], dtype=np.int8)
        start_seq = start_seq.reshape((1, len(start_seq)))
        start_seq = np.repeat(start_seq, no_samples, axis=0)
        random_sequences = np.random.choice(char_indices, size=(no_samples, len_canary), replace=True)
        perplexities = np.zeros(no_samples)
        for i in range(random_sequences.shape[1]):
            # add a random char to the sequences to predict them
            start_seq = np.insert(start_seq, start_seq.shape[1], random_sequences[:, i], axis=1)
            # extract proba of inserted chars
            start_seq_pred = model.predict(start_seq, batch_size=1000, verbose=1)[range(no_samples), random_sequences[:, i]]
            perplexities -= np.log2(start_seq_pred)
        perplexities = np.array(perplexities)
        if plot:
            _, bins, _ = plt.hist(perplexities, bins=1000, density=True)
            ae, loc, scale = skewnorm.fit(perplexities)
            print('Skewnorm-fit parameters: {0:.3f} - {1:.3f} - {2:.3f}'.format(ae, loc, scale))
            sn = skewnorm(ae, loc, scale)
            x = np.linspace(0, max(bins), 500)
            plt.plot(x, sn.pdf(x), linewidth=5.0, label="Skewnorm-fit")
            plt.xlabel('Log Perplexity')
            plt.ylabel('Relative Frequency')
            plt.legend()
            plt.show()
        return perplexities

    def approx_exposure(self, perplexity_values, perplexities, only_digits=False):
        if type(perplexity_values) is not list:
            perplexity_values = [perplexity_values]
        if only_digits:
            numbers = np.unique([d for d in self.canary_number])
            char_indices = [self.char2idx[n] for n in numbers]
        else:
            char_indices = list(self.idx2char.keys())
        n_combinations = len(char_indices) ** len(self.canary_number)  # total number of combinations possible
        R = np.log2(float(n_combinations))  # if n_combinations is large np.log2 needs needs float
        ae, loc, scale = skewnorm.fit(perplexities)
        sn = skewnorm(ae, loc, scale)
        quantiles = [sn.cdf(pv) for pv in perplexity_values]
        exposures = [-np.log2(q) for q in quantiles]
        for i in range(len(perplexity_values)):
            print('Results for {}'.format(perplexity_values[i]))
            print('{}% of all sequences are more likely than the given one.'.format(quantiles[i] * 100))
            print('Log(|R|) is {}'.format(R))
            print('The exposure of the sequence is {}'.format(exposures[i]))
        return exposures

    # method that prints index arrays as words
    # input array should be of shape (batch_size, seq_length, 1)
    def indices_to_words(self, idx_array):
        if len(idx_array.shape) == 1:
            idx_array = [idx_array]
        for idx, arr in enumerate(idx_array):
            print(f'idx {idx}')
            s_arr = ''.join([self.idx2char[i] for i in arr.flatten()])
            print(s_arr)