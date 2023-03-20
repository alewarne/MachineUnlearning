import os
import time
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from pathlib import Path


# returns a list of (x,y) tuples where x is data and y is labels. The data is specified by a doc path (to a file
# containing the feature vectors) and a split path containing files with filenames for training, testing, validation.
# since there are several splits for the dataset, the index spcifies which split to choose
def get_train_test_data(min_doc_count, test_set_size):
    path_to_benign_shas = os.path.join(sha_folder, benign_sha_name)
    path_to_mal_shas = os.path.join(sha_folder, malicious_sha_name)
    benign_filepaths = [os.path.join(doc_path, name) for name in open(path_to_benign_shas).read().splitlines()]
    malicious_filepaths = [os.path.join(doc_path, name) for name in open(path_to_mal_shas).read().splitlines()]
    all_paths = benign_filepaths+malicious_filepaths
    vec = CountVectorizer(input='filename', token_pattern='.+', lowercase=False, min_df=min_doc_count)
    start_time = time.time()
    data = vec.fit_transform(all_paths)
    end_time = time.time()
    print('Count Vectorizer took {} seconds'.format(end_time-start_time))
    labels = np.array([-1]*len(benign_filepaths) + [1]*len(malicious_filepaths))
    # clean data
    if min_doc_count > 1:
        # if min_doc_count is greater than 1 possible duplicate lines can occur in the data
        data, labels = clean_data_sp(data, labels)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_set_size, random_state=42)
    return (x_train, y_train), (x_test, y_test), vec.vocabulary_


def clean_data_sp(data, labels):
    """
    Takes a sparse csr_matrix as input and removes lines with only zeros and duplicate entries. Removes the
     corresponding lines in labels as well
    """
    assert data.shape[0] == labels.shape[0]
    # remove rows with only zeros
    remaining_axes = data.getnnz(axis=1) > 0
    if not remaining_axes.all():
        print('Found {} empty lines. Deleting.'.format(len(remaining_axes)-np.sum(remaining_axes)))
    data = data[remaining_axes]
    labels = labels[remaining_axes]
    # remove duplicate rows
    all_column_indices = [data[i].indices.tolist() for i in range(data.shape[0])]
    unique_row_indices, unique_columns = [], []
    for row_idx, indices in enumerate(all_column_indices):
        if indices not in unique_columns:
            unique_columns.append(indices)
            unique_row_indices.append(row_idx)
    if len(unique_row_indices) != data.shape[0]:
        diff = data.shape[0] - len(unique_row_indices)
        print('Original number of samples:{}'.format(data.shape[0]))
        print('Duplicates removed:{}'.format(diff))
        print('Samples remaining:{}'.format(len(unique_row_indices)))
    return data[unique_row_indices], labels[unique_row_indices]


def quick_test(train_data, test_data):
    from sklearn.svm import SVC
    svm = SVC()
    print('Training SVM ...')
    start_time = time.time()
    svm.fit(train_data[0], train_data[1])
    end_time = time.time()
    print('Fitting the SVM took {} seconds'.format(end_time-start_time))
    y_pred = svm.predict(test_data[0])
    acc = accuracy_score(test_data[1], y_pred)
    prec = precision_score(test_data[1], y_pred)
    recall = recall_score(test_data[1], y_pred)
    print(f'Acc: {acc}, Precision: {prec}, Recall: {recall}')


def extract_urls(voc):
    l = [w for w in voc if 'url' in w]
    with open('relevant_features.txt', 'w') as f:
        for url in l:
            print(url, file=f)


if __name__ == "__main__":
    doc_path = os.path.join(str(Path.home()), 'drebin_dataset/drebin-public/feature_vectors/')
    sha_folder = 'sha_lists/'
    benign_sha_name = 'drebin_benign_new.shas'
    malicious_sha_name = 'drebin_malicious_new.shas'
    min_doc_count = 100  # min number of apps a feature has to appear in
    test_set_size = 0.2
    train_data, test_data, voc = get_train_test_data(min_doc_count, test_set_size)
    sp.save_npz('x_train.npz', train_data[0])
    np.save('y_train.npy', train_data[1])
    sp.save_npz('x_test.npz', test_data[0])
    np.save('y_test.npy', test_data[1])
    pkl.dump(voc, open('voc.pkl', 'wb'))
    extract_urls(voc)
    print('{} training samples, {} test samples, {} total'.format(train_data[0].shape[0], test_data[0].shape[0],
                                                                 train_data[0].shape[0] + test_data[0].shape[0]))
    print('Number of features: {}'.format(train_data[0].shape[1]))
    quick_test(train_data, test_data)
