import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm
from Unlearner.DNNUnlearner import DNNUnlearner

N_ROWS = 32
N_CHANNELS = 3


# class to "unlearn" label changes
class CNNUnlearner(DNNUnlearner):
    def __init__(self, train, test, valid, weight_path=None, lambda_=1e-5):
        self.x_train = train[0]
        self.y_train = train[1]
        self.x_test = test[0]
        self.y_test = test[1]
        self.x_valid = valid[0]
        self.y_valid = valid[1]
        self.lambda_ = lambda_
        self.n = self.x_train.shape[0]
        self.dim = self.x_train.shape[1]
        self.model = self.get_network(weight_path)

    def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
        n_filters = [32, 32, 64, 64, 128, 128]
        conv_params = dict(activation='relu', kernel_size=3, kernel_initializer='he_uniform', padding='same')

        model = Sequential()
        # 1st VGG block
        model.add(Conv2D(filters=n_filters[0], input_shape=(N_ROWS, N_ROWS, N_CHANNELS), **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=n_filters[1], **conv_params))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        # 2nd VGG block
        model.add(Conv2D(filters=n_filters[2], **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=n_filters[3], **conv_params))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        # 3rd VGG block
        model.add(Conv2D(filters=n_filters[4], **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=n_filters[5], **conv_params))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # dense and final layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(units=10, activation='softmax'))

        if optimizer == 'Adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate, amsgrad=True, epsilon=0.1),
                          loss=categorical_crossentropy, metrics='accuracy')
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate), loss=categorical_crossentropy, metrics='accuracy')
        if weight_path is not None:
            model.load_weights(weight_path)
        print(model.summary())
        return model

    # returns copy of train/test/valid set in which certain coordinates can be set to zero
    # indices_to_delete is a numpy array of shape (N,2) where each row represents a coordinate to be deleted.
    def get_data_copy(self, data_name, indices_to_delete, **kwargs):
        assert data_name in ['train', 'test', 'valid']
        affected_samples = kwargs['affected_samples'] if 'affected_samples' in kwargs else None
        if data_name == 'train':
            data_cpy = self.x_train.copy()
        elif data_name == 'test':
            data_cpy = self.x_test.copy()
        else:
            data_cpy = self.x_valid.copy()
        if len(indices_to_delete) > 0:
            if affected_samples is not None:
                for affected_idx in affected_samples:
                    data_cpy[affected_idx, indices_to_delete[:, 0], indices_to_delete[:, 1], :] = 0
            else:
                data_cpy[:, indices_to_delete[:, 0], indices_to_delete[:, 1], :] = 0
        return data_cpy

    # returns indices of samples in training_data where the features corresponding to 'indices_to_delete' are not zero
    # indices_to_delete is a numpy array of shape (N,2) where each row represents one coordinate to check
    def get_relevant_indices(self, indices_to_delete):
        relevant_rows = set()
        # sum all channels and take all indices where at least one of the coordinates is not zero
        x_train_channel_sum = np.sum(self.x_train, axis=3)
        for coordinate in indices_to_delete:
            nz = x_train_channel_sum[:, coordinate[0], coordinate[1]].nonzero()[0]
            relevant_rows = relevant_rows.union(set(nz))
        return list(relevant_rows)

    # calculates influences of pixels when setting them to 0 during training via influece functions.
    # deletion_size specifies size of pixels to set to 0. Deletion_size=1 sets 1 pixel at a time to zero whereas
    # deletion_size=4 sets blocks of 4x4 to zero
    def explain_prediction(self, x, y, deletion_size=1, **kwargs):
        assert N_ROWS % deletion_size == 0
        batch_size = 500 if 'batch_size' not in kwargs else kwargs['batch_size']
        rounds = 1 if 'rounds' not in kwargs else kwargs['rounds']
        scale = 75000 if 'scale' not in kwargs else kwargs['scale']
        damping = 1e-2 if 'damping' not in kwargs else kwargs['damping']
        verbose = False if 'verbose' not in kwargs else kwargs['verbose']
        relevances = np.zeros_like(x)
        grad_x = self.get_gradients(x, y)
        H_inv_grad_x, diverged = self.get_inv_hvp_lissa(grad_x, batch_size, scale, damping, verbose, rounds)
        print('Calculating influence for {} rows'.format(N_ROWS/deletion_size))
        for i in tqdm(range(0, N_ROWS, deletion_size)):
            for j in range(0, N_ROWS, deletion_size):
                square_to_delete = np.array(np.meshgrid(range(i, i+deletion_size),
                                                        range(j, j+deletion_size))).T.reshape(-1, 2)
                relevant_indices = self.get_relevant_indices(indices_to_delete=square_to_delete)
                if len(relevant_indices) == 0:
                    relevances[0, i:i+deletion_size, j:j+deletion_size] = 0
                    continue
                else:
                    max_indices = 256
                    if len(relevant_indices) > max_indices:
                        relevant_indices = np.random.choice(relevant_indices, max_indices, replace=False)
                    x_train_zero = self.x_train[relevant_indices].copy()
                    x_train_zero[:, i:i+deletion_size, j:j+deletion_size] = 0
                    d_L_z = self.get_gradients(self.x_train[relevant_indices], self.y_train[relevant_indices])
                    d_L_z_delta = self.get_gradients(x_train_zero, self.y_train[relevant_indices])
                    diff = [dLd - dL for dLd, dL in zip(d_L_z_delta, d_L_z)]
                    loss_diff_per_param = [np.sum(d*Hd) for d, Hd in zip(diff, H_inv_grad_x)]
                    relevances[0, i:i+deletion_size, j:j+deletion_size] = np.sum(loss_diff_per_param)/self.n
        return relevances, diverged


class CNNUnlearnerMedium(CNNUnlearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
        n_filters = [64, 64, 128, 128, 256, 256]
        kernel_size = 3
        model = Sequential()
        model.add(Conv2D(filters=n_filters[0], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same', input_shape=(N_ROWS, N_ROWS, N_CHANNELS)))
        model.add(Conv2D(filters=n_filters[1], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Conv2D(filters=n_filters[2], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(Conv2D(filters=n_filters[3], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Conv2D(filters=n_filters[4], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(Conv2D(filters=n_filters[5], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_regularizer=L2(self.lambda_)))
        model.add(Dropout(0.6))
        # final dense layer
        model.add(Dense(units=10, activation='softmax', kernel_regularizer=L2(self.lambda_)))
        if optimizer == 'Adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
                          loss=categorical_crossentropy, metrics='accuracy')
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate), loss=categorical_crossentropy, metrics='accuracy')
        if weight_path is not None:
            model.load_weights(weight_path)
        # print(model.summary())
        return model


class CNNUnlearnerSmall(CNNUnlearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
        n_filters = [64, 64, 128, 128]
        kernel_size = 3
        model = Sequential()
        model.add(Conv2D(filters=n_filters[0], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same', input_shape=(N_ROWS, N_ROWS, N_CHANNELS)))
        model.add(Conv2D(filters=n_filters[1], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Conv2D(filters=n_filters[2], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(Conv2D(filters=n_filters[3], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=L2(self.lambda_)))
        model.add(Dropout(0.6))
        # final dense layer
        model.add(Dense(units=10, activation='softmax', kernel_regularizer=L2(self.lambda_)))
        if optimizer == 'Adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
                          loss=categorical_crossentropy, metrics='accuracy')
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate), loss=categorical_crossentropy, metrics='accuracy')
        if weight_path is not None:
            model.load_weights(weight_path)
        print(model.summary())
        return model


class CNNUnlearnerAvgPooling(CNNUnlearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
        n_filters = [64, 64, 128, 128]
        kernel_size = 3
        model = Sequential()
        model.add(Conv2D(filters=n_filters[0], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same', input_shape=(N_ROWS, N_ROWS, N_CHANNELS)))
        model.add(Conv2D(filters=n_filters[1], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Conv2D(filters=n_filters[2], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(Conv2D(filters=n_filters[3], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=L2(self.lambda_)))
        model.add(Dropout(0.6))
        # final dense layer
        model.add(Dense(units=10, activation='softmax', kernel_regularizer=L2(self.lambda_)))
        if optimizer == 'Adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
                          loss=categorical_crossentropy, metrics='accuracy')
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate), loss=categorical_crossentropy, metrics='accuracy')
        if weight_path is not None:
            model.load_weights(weight_path)
        print(model.summary())
        return model
