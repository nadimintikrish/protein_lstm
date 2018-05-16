class EmbeddingEncoder:
    def __init__(self):
        self.aa_set = set()
        self.embed_encoder = {}
        self.decode_dict = {}

    def find_unique_amino_acids(self, seq_series):
        for i in seq_series:
            self.aa_set = self.aa_set.union(set(list(i)))
        return self.aa_set

    def create_embed_encoder(self):
        aa_place = 0
        for aa in self.aa_set:
            self.embed_encoder[aa] = aa_place
            self.decode_dict[aa_place] = aa
            aa_place += 1

    def apply_embed_encoding(self, seq):
        embed_array = list(map(self.create_embed_encoder, seq))
        return embed_array

    def get_aa_set(self):
        return self.aa_set
