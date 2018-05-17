from utilities.OneHotEncoder import OneHotEncoder
import numpy as np
import pandas as pd


class DataPreProcessor:

    def __init__(self, count_dict, embed_encoder, one_hot_encoder):
        self.count_dict = count_dict  # add count-dict
        self.least_preferred = []
        self.X_list_seq = []
        self.y = pd.DataFrame()
        self.protein_categories = []
        self.one_hot_encoder = one_hot_encoder
        self.embed_encoder = embed_encoder

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
        self.protein_categories = list(self.y.columns)
        return self.X_list_seq

    def process_seqs_to_embeddings(self, X, y, max_length):
        apply_embed_encoding = self.embed_encoder.apply_embed_encoding
        temp_x = list(map(apply_embed_encoding, X))
        for i in range(len(temp_x)):
            rem_len = max_length - len(temp_x[i])
            if rem_len > 0:
                temp_x[i].extend([0] * rem_len)
        temp_y = pd.get_dummies(y)
        self.protein_categories = list(temp_y.columns)
        self.X_list_seq = np.array(temp_x)
        self.y = np.array(temp_y)

    def process_seqs_to_arrays(self, X, y, max_length):
        apply_one_hot_encoding = self.one_hot_encoder.apply_one_hot_encoding
        temp_x = list(map(apply_one_hot_encoding, X))
        for i in range(len(temp_x)):
            rem_len = max_length - len(temp_x[i])
            if rem_len > 0:
                temp_x[i].extend([[0] * len(self.one_hot_encoder.aa_set)] * rem_len)
        temp_y = pd.get_dummies(y)
        self.protein_categories = list(temp_y.columns)
        self.X_list_seq = np.array(temp_x)
        self.y = np.array(temp_y)

    def decode_categories(self, encoded_array):
        place = np.argmax(encoded_array)
        return self.protein_categories[place]

    def batch_generator(self, n=1):
        l = len(self.X_list_seq)
        for ndx in range(0, l, n):
            sliced = self.X_list_seq[ndx:min(ndx + n, l)]
            y_sliced = self.y[ndx:min(ndx + n, l)]  # to-test
            max_len = max(map(len, sliced))
            for i in range(len(sliced)):
                rem_len = max_len - len(sliced[i])
                if rem_len > 0:
                    sliced[i].extend([[0] * len(self.one_hot_encoder.aa_set)] * rem_len)  # to optimize
            yield np.array(sliced), np.array(y_sliced)

    def get_protein_categoeies(self):
        return self.protein_categories

    def count_dict_keys_as_list(self):
        return list(self.count_dict.keys())

    def get_embed_class(cls):
        return cls.embed_encoder

    def get_x(self):
        return self.X_list_seq

    def get_y(self):
        return self.y
