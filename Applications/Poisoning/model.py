from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD


CIFAR_SHAPE = (32, 32, 3)


def get_VGG_CIFAR10(input_shape=CIFAR_SHAPE, weight_path=None, lr_init=0.001, dense_units=512, sgd=False):
    n_filters = [128, 128, 128, 128, 128, 128]
    conv_params = dict(activation='relu', kernel_size=3,
                       kernel_initializer='he_uniform', padding='same')

    model = Sequential()
    # VGG block 1
    model.add(Conv2D(filters=n_filters[0], input_shape=input_shape, **conv_params))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=n_filters[1], **conv_params))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    # VGG block 2
    model.add(Conv2D(filters=n_filters[2], **conv_params))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=n_filters[3], **conv_params))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    # VGG block 3
    model.add(Conv2D(filters=n_filters[4], **conv_params))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=n_filters[5], **conv_params))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # dense and final layers
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(Dense(units=10, activation='softmax'))

    # compile model, optionally load weights
    if sgd:
        model.compile(optimizer=SGD(learning_rate=lr_init), loss=categorical_crossentropy, metrics='accuracy')
    else:
        model.compile(optimizer=Adam(learning_rate=lr_init, amsgrad=True),
                      loss=categorical_crossentropy, metrics='accuracy')
    print(model.summary())
    if weight_path is not None:
        model.load_weights(weight_path)
    return model
