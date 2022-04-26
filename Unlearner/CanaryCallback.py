import tensorflow as tf
import numpy as np

class CanaryCallback(tf.keras.callbacks.Callback):
    """After each training epoch, generate a sequence based on the canary sentence"""

    def __init__(self, canary_start, int_to_char, canary_number, frequency=10):
        super(CanaryCallback, self).__init__()
        self.canary_start = canary_start
        self.int_to_char = int_to_char
        self.char_to_int = {v: k for k,v in self.int_to_char.items()}
        self.canary_number = canary_number
        self.number_char_indices = np.unique([self.char_to_int[c] for c in self.canary_number])
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        chars_to_predict = 20
        if epoch % self.frequency == 0:
            start_seq = np.array([self.char_to_int[s] for s in self.canary_start])
            start_seq = start_seq.reshape((1, len(start_seq), 1))
            prediction_str, dist_str = '', ''
            digit_probas_str = ' - '.join(['{}:{{:.4f}}'.format(self.int_to_char[i]) for i in self.number_char_indices])
            # generate characters
            for i in range(chars_to_predict):
                index_distribution = self.model.predict(start_seq)
                char_index = np.argmax(index_distribution)
                prediction_str += self.int_to_char[char_index]
                start_seq = np.append(start_seq, char_index.reshape(1, 1, 1), axis=1)
                start_seq = start_seq[:, 1:start_seq.shape[1] + 1, :]
                # save distribution of numbers after the canary
                if i < len(self.canary_number):
                    digit_probas = index_distribution[0, self.number_char_indices]
                    dist_str += 'Step {} : '.format(i) + digit_probas_str.format(*digit_probas) + '\n'
            print("\nSeed:")
            print("\"", self.canary_start, "\"")
            print('...')
            print(prediction_str)
            print("Number distribution:")
            print(dist_str)

    def on_predict_end(self, logs=None):
        self.on_epoch_end(0)

    # for some reasons this methods deletes the history. dunno why
    #def on_train_end(self, logs=None):
    #    self.on_epoch_end(0)
