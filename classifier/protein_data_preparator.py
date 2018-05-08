import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_pdb_data_no_seq = pd.read_csv('pdb_data_no_dups.csv')
df_pdb_seq = pd.read_csv('pdb_data_seq.csv')

# filtering out only protein structures
df_protein_seq = df_pdb_seq[(df_pdb_seq['macromoleculeType']=='Protein')]

# joining two data sets for protein Seq - classification

df_protein_final = df_protein_seq.merge(df_pdb_data_no_seq)

df_protein_seq_final = df_protein_final[['sequence','classification']]

# drop na values in the data frame

df_protein_seq_final = df_protein_seq_final.dropna()
