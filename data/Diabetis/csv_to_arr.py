import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split


def csv_to_arr(csv_path):
    df = pd.read_csv(csv_path, sep=',')
    voc = {c:i for c,i in zip(df.columns, range(len(df.columns)))}
    arr = df.values
    x = arr[:, :-1]
    y = arr[:, -1]
    y = 2*y-1  # map labels from {0,1} to {-1,1}
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train_scaled, x_test_scaled = minmax_scale(x_train), minmax_scale(x_test)
    max_l2_norm = np.max(np.sqrt(np.sum(x_train_scaled**2, axis=1)))
    print(f'Max l2 norm of data: {max_l2_norm}')
    return (x_train_scaled, y_train), (x_test_scaled, y_test), voc


def quick_test(train_data, test_data):
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(max_iter=500, C=1, fit_intercept=False)
    LR.fit(train_data[0], train_data[1])
    y_pred = LR.predict(test_data[0])
    acc = len(np.where(y_pred == test_data[1])[0])/y_pred.shape[0]
    print(f'Acc: {acc}')


if __name__ == '__main__':
    csv_path = 'diabetes.csv'
    train_data, test_data, voc = csv_to_arr(csv_path)
    quick_test(train_data, test_data)
    np.save('x_train.npy', train_data[0])
    np.save('y_train.npy', train_data[1])
    np.save('x_test.npy', test_data[0])
    np.save('y_test.npy', test_data[1])
    pkl.dump(voc, open('voc.pkl', 'wb'))
    print('{} training samples, {} test samples, {} total'.format(train_data[0].shape[0], test_data[0].shape[0],
                                                                    train_data[0].shape[0] + test_data[0].shape[0]))
    print('Number of features: {}'.format(train_data[0].shape[1]))
