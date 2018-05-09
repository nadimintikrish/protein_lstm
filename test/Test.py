# from utilities.DataPreProcessor import DataPreProcessor
#
# test = DataPreProcessor()
# test_X = [[[1, 2, 3], [2, 4, 5], [5, 6, 7]], [[1, 5, 6], [7, 8, 9], [0, 1, 2], [4, 6, 2]], [[1, 4, 1], [2, 4, 5]],
#           [[0, 2, 5]], [[0, 5, 5], [1, 1, 1]]]
# test_y = [[1, 0], [0, 1], [1, 0], [0, 1], [0, 1]]
#
# for t1, t2 in test.batch_gen(test_X, test_y, 2):
#     print(t1, t2)

from random import randint
import numpy as np
from model.LSTMModel import LSTMModel
## generate seq of numbers

def generate_sequence(length,n_features):
    return [randint(0,n_features-1) for _ in range(length)]

## one hot encode sequence

def one_hot_encode(sequence, n_features):
    encoding = []
    for value in sequence:
        vector = [0 for _ in range(n_features)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)

def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]

# generate one example for an lstm
def generate_example(length, n_features, out_index):
# generate sequence
    sequence = generate_sequence(length, n_features)
    # one hot encode
    encoded = one_hot_encode(sequence, n_features)
    # reshape sequence to be 3D
    X = encoded.reshape((1, length, n_features))
    # select output
    y = encoded[out_index].reshape(1, n_features)
    return X, y

X,y = generate_example(25,100,2)
print(X.shape)
print(y.shape)


lstm_model = LSTMModel(5,10)
print(lstm_model.define_model(25,2))

for i in range(1000):
    X,y = generate_example(5,10,2)
    lstm_model.model.fit(X,y,epochs=1,verbose=2)