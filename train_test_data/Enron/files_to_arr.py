import os
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pickle as pkl


def get_train_test_data(ham_path, spam_path, enron_indices, min_doc_count, binary=False):
    filenames = [ham_path.format(i) for i in enron_indices] + [spam_path.format(i) for i in enron_indices]
    filelist = [os.path.join(p,n) for p in filenames for n in os.listdir(p)]
    labels = np.array([1 if 'spam' in f else -1 for f in filelist])
    vectorizer = TfidfVectorizer(input='filename', encoding='latin-1', min_df=min_doc_count, binary=binary)
    vectorizer = CountVectorizer(input='filename', encoding='latin-1', min_df=min_doc_count, binary=binary)
    x = vectorizer.fit_transform(filelist)
    # normalize data
    x_normed = normalize(x, norm='l2', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x_normed, labels, test_size=0.2, random_state=0)
    return (x_train, y_train), (x_test, y_test), vectorizer.vocabulary_


def quick_test(train_data, test_data):
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(max_iter=500)
    LR.fit(train_data[0], train_data[1])
    y_pred = LR.predict(test_data[0])
    acc = len(np.where(y_pred == test_data[1])[0])/y_pred.shape[0]
    print(f'Acc: {acc}')


if __name__ == '__main__':
    ham_path = 'raw/enron{}/ham'
    spam_path = 'raw/enron{}/spam'
    enron_indices = [1, 2, 3, 4, 5, 6]
    min_doc_count = 100
    binary = True
    train_data, test_data, voc = get_train_test_data(ham_path, spam_path, enron_indices, min_doc_count, binary)
    sp.save_npz('x_train.npz', train_data[0])
    np.save('y_train.npy', train_data[1])
    sp.save_npz('x_test.npz', test_data[0])
    np.save('y_test.npy', test_data[1])
    pkl.dump(voc, open('voc.pkl', 'wb'))
    print('{} training samples, {} test samples, {} total'.format(train_data[0].shape[0], test_data[0].shape[0],
                                                                  train_data[0].shape[0] + test_data[0].shape[0]))
    print('Number of features: {}'.format(train_data[0].shape[1]))
    quick_test(train_data, test_data)
