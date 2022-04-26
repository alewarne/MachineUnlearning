import numpy as np
import re
from tensorflow.keras.utils import to_categorical
from nltk.tokenize import word_tokenize, sent_tokenize


def load_data(filename, seq_length, canary, canary_insertions, int2char=None):
    np.random.seed(42)
    # load ascii text, insert canary and covert to lowercase
    raw_text = open(filename, 'r', encoding='utf-8').read()[265:]
    raw_text = insert_canary(raw_text, canary, canary_insertions)
    raw_text = raw_text.lower()
    chars = sorted(list(set(raw_text)))
    # create mapping of unique chars to integers, and a reverse mapping
    if int2char is not None:
        char2int = {v:k for k,v in int2char.items()}
    else:
        char2int = dict((c, i) for i, c in enumerate(chars))
        int2char = {v:k for k,v in char2int.items()}
    n_chars = len(raw_text)
    # summarize the loaded data
    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char2int[char] for char in seq_in])
        dataY.append(char2int[seq_out])
    n_patterns = len(dataX)
    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # one hot encode the output variable
    y = to_categorical(dataY)
    return X, y, int2char


def insert_canary(text, canary, n_insertions):
    if n_insertions == 0:
        return text
    canary_len = 4  # 2 newlines + 2 spaces
    breaks = [m.start() for m in re.finditer('\n\n  \w', text)]
    insertion_points = sorted(np.random.choice(breaks, n_insertions, replace=False))
    new_text = ''
    for idx in range(len(insertion_points)):
        point_pre = insertion_points[idx-1]+canary_len if idx != 0 else 0
        point_last = insertion_points[idx]+canary_len
        new_text += text[point_pre:point_last] + canary
    new_text += text[point_last:]
    return new_text


def get_words(filename):
    # load ascii text and covert to lowercase
    raw_text = open(filename, 'r', encoding='utf-8').read()[265:]
    raw_text = raw_text.lower()
    sentences = sent_tokenize(raw_text)
    words_l = [word_tokenize(s) for s in sentences]
    all_words = [w for ws in words_l for w in ws]
    n_words = len(all_words)
    words, word_count = np.unique(all_words, return_counts=True)
    word_counts = {w:wc for w,wc in zip(words, word_count)}
    #word_counts_rel = {k:v/n_words for k,v in word_counts.items()}
    #word_counts_rel_sorted = {k:v for k,v in sorted(word_counts_rel.items(), key= lambda x: x[1], reverse=True)}
    #random_words = np.random.choice(all_words, 10, replace=False)
    return words, word_counts