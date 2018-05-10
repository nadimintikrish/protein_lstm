import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utilities.OneHotEncoder import OneHotEncoder
from utilities.DataPreProcessor import DataPreProcessor

df_pdb_data_no_seq = pd.read_csv('C:/Krishna/WorkArea/protein_lstm/resources/pdb_data_no_dups.csv')
df_pdb_seq = pd.read_csv('C:/Krishna/WorkArea/protein_lstm/resources/pdb_data_seq.csv')

print("..Read data files..")

# filtering out only protein structures
df_protein_seq = df_pdb_seq[(df_pdb_seq['macromoleculeType'] == 'Protein')]

# joining two data sets for protein Seq - classification

df_protein_final = df_protein_seq.merge(df_pdb_data_no_seq)

df_protein_seq_final = df_protein_final[['sequence', 'classification']]

# drop na values in the data frame

df_protein_seq_final = df_protein_seq_final.dropna()

print(df_protein_seq_final.head())

print("filtered data with NA values")
##One_hot encoding

one_hot_encoder = OneHotEncoder()

one_hot_encoder.find_unique_amino_acids(df_protein_seq_final['sequence'])
print("Unique Amino Acid Set...")

## creates a encoded dictionaryfor each Amino Acid
one_hot_encoder.create_one_hot_dict()

data_pre_processor = DataPreProcessor(one_hot_encoder,
                                      df_protein_seq_final['classification']
                                      .value_counts().to_dict())
## delete least preferred
data_pre_processor.del_least_preferred()

df_protein_seq_final_for_modeling = \
    df_protein_seq_final[df_protein_seq_final['classification']
        .isin(data_pre_processor.count_dict_keys_as_list())]

df_protein_seq_final_for_modeling = \
    df_protein_seq_final_for_modeling.reset_index(drop=True)
print("final Shape of the DataSet {}".format(df_protein_seq_final_for_modeling.shape))

X_sequences = df_protein_seq_final_for_modeling['sequence']
y_label = df_protein_seq_final_for_modeling['classification']

data_pre_processor.process_all_seqs(X_sequences[:15], y_label[:15])
print("Getting Processed Seqs")
print(data_pre_processor.get_x())
print("getting labels")
print(data_pre_processor.get_y())
print("getting protein categories")
print(data_pre_processor.get_protein_categoeies())