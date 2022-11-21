from tensorflow.keras.utils import Sequence as seq
import numpy as np


class DrebinDataGenerator(seq):
    def __init__(self, data, labels, batch_size, shuffle=True, class_weights=None):
        np.random.seed(42)
        assert data.shape[0] == labels.shape[0]
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        if shuffle:
            self.on_epoch_end()
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = {0:1, 1:1}

    def on_epoch_end(self):
        if self.shuffle:
            new_indices = np.random.choice(range(self.data.shape[0]), self.data.shape[0], replace=False)
            self.data = self.data[new_indices]
            self.labels = self.labels[new_indices]

    def __len__(self):
        return int(np.ceil(self.data.shape[0]/self.batch_size))

    def __getitem__(self, idx):
        data_batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size, :].toarray()
        label_batch = [self.labels[idx*self.batch_size:(idx + 1)*self.batch_size]]
        classes = np.argmax(self.labels[idx*self.batch_size:(idx + 1)*self.batch_size], axis=1)
        samples_weights = np.array([self.class_weights[0] if c==0 else self.class_weights[1] for c in classes])
        return data_batch, label_batch, samples_weights
