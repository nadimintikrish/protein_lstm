from utilities.OneHotEncoder import OneHotEncoder
import numpy as np


class DataPreProcessor:
    def __init__(self):
        self.count_dict = {}
        self.least_preferred = []
        self.X_list_seq = []
        self.y = []
        self.one_hot_encoder = OneHotEncoder()

    def list_least_preferred(self):
        self.least_preferred = [i for i in self.count_dict if i < 500]
        return self.least_preferred

    def del_least_preferred(self):
        for i in self.list_least_preferred():
            del self.count_dict[i]

    def process_all_seqs(self, X, y):
        self.X_list_seq = list(map(lambda x:
                                   self.one_hot_encoder.apply_one_hot_encoding(x), X))
        self.y = list(y)
        return self.X_list_seq

    def batch_generator(self, n=1):
        l = len(self.X_list_seq)
        for ndx in range(0, l, n):
            sliced = self.X_list_seq[ndx:min(ndx + n, l)]
            y_sliced = self.y[ndx:min(ndx + n, l)]  # to-test
            max_len = max(map(lambda x: len(x), sliced))
            for i in range(len(sliced)):
                rem_len = max_len - len(sliced[i])
                if rem_len > 0:
                    sliced[i].extend([[0] * 3] * rem_len)
            yield np.array(sliced), np.array(y_sliced)
