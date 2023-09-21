# Made by Naman Garg
# Contact : <12112061@nitkkr.ac.in>

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../music_data_source_cleaned.csv")

# I will take only first 4 attributes, that is
# chroma_stft_mean,chroma_stft_var,rms_mean,rms_var
df = dataset.iloc[:, 1:5]

X = df.values

dist_matrix = sch.distance.pdist(X)

# Perform hierarchical clustering
linkage_matrix = sch.linkage(dist_matrix, method='complete')

# Plot dendrogram
fig, ax = plt.subplots(figsize=(12, 12))
sch.dendrogram(linkage_matrix, ax=ax)

# Cut dendrogram to get clusters
clusters = sch.fcluster(linkage_matrix, t=7, criterion='distance')

plt.show()
