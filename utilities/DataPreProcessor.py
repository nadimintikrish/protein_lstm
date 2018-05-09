from utilities.OneHotEncoder import OneHotEncoder
import numpy as np
import pandas as pd


class DataPreProcessor:

    def __init__(self, one_hot_encoder, count_dict):
        self.count_dict = count_dict  # add count-dict
        self.least_preferred = []
        self.X_list_seq = []
        self.y = pd.DataFrame()
        self.one_hot_encoder = one_hot_encoder

    def list_least_preferred(self):
        self.least_preferred = [i for i in self.count_dict if self.count_dict[i] < 500]
        return self.least_preferred

    def del_least_preferred(self):
        for i in self.list_least_preferred():
            del self.count_dict[i]

    def process_all_seqs(self, X, y):
        apply_one_hot_encoding = self.one_hot_encoder.apply_one_hot_encoding
        self.X_list_seq = list(map(apply_one_hot_encoding, X))
        self.y = pd.get_dummies(y)
        return self.X_list_seq

    def batch_generator(self, n=1):
        l = len(self.X_list_seq)
        for ndx in range(0, l, n):
            sliced = self.X_list_seq[ndx:min(ndx + n, l)]
            y_sliced = self.y[ndx:min(ndx + n, l)]  # to-test
            max_len = max(map(len, sliced))
            for i in range(len(sliced)):
                rem_len = max_len - len(sliced[i])
                if rem_len > 0:
                    sliced[i].extend([[0] * 3] * rem_len)
            yield np.array(sliced), np.array(y_sliced)

    @staticmethod
    def batch_gen(X, y, n=1):
        l = len(X)
        for ndx in range(0, l, n):
            sliced = X[ndx:min(ndx + n, l)]
            y_sliced = y[ndx:min(ndx + n, l)]  # to-test
            max_len = max(map(len, sliced))
            for i in range(len(sliced)):
                rem_len = max_len - len(sliced[i])
                if rem_len > 0:
                    sliced[i].extend([[0] * 3] * rem_len)
            yield np.array(sliced), np.array(y_sliced)

    def count_dict_keys_as_list(self):
        return list(self.count_dict.keys())

    def get_x(self):
        return self.X_list_seq

    def get_y(self):
        return self.y
