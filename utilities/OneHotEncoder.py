class OneHotEncoder:
    def __init__(self):
        self.aa_set = set()
        self.aa_one_hot_dict = {}

    def find_unique_amino_acids(self, seq_series):
        for i in seq_series:
            self.aa_set = self.aa_set.union(set(list(i)))
        return self.aa_set

    def create_one_hot_dict(self):
        aa_place = 0
        for aa in self.get_aa_set():
            self.aa_one_hot_dict[aa] = [0] * len(self.aa_set)
            self.aa_one_hot_dict[aa][aa_place] = 1
            aa_place += 1

    def apply_one_hot_encoding(self, seq):
        one_hot_array = [self.aa_one_hot_dict[i] for i in seq]
        return one_hot_array

    def get_aa_set(self):
        return self.aa_set
