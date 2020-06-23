# libraris
import pickle
import gzip
import numpy as np

def train_data():
    tr_d, te_d = load_data
    train_images = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    train_labels = [vectorized_result(y) for y in tr_d[1]]
    train_data = zip(train_images, train_labels)
    return train_data

def test_data():
    tr_d, te_d = load_data()
    test_images = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_images, te_d[1])
    return test_data

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, test_data = pickle.load(f)
    f.close()
    return (training_data, test_data)

def vectorized_result(j):
    # Return a 10-dimensional unit vector with a 1.0 in the jth
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e  
