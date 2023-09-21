# Made by Naman Garg
# Contact : <12112061@nitkkr.ac.in>

# Requires fuzzy-c-means package. Use
# pip install fuzzy-c-means
# to install it.

import pandas as pd
from fcmeans import FCM
from matplotlib import pyplot as plt

dataset = pd.read_csv("../music_data_source_cleaned.csv")

# I will take only first 4 attributes, that is
# chroma_stft_mean,chroma_stft_var,rms_mean,rms_var
df = dataset.iloc[:, 1:5]

x = df.values

# We are dividing our music files into 7 clusters
fcm = FCM(n_clusters=7)
fcm.fit(x)

fcm_centers = fcm.centers
fcm_labels = fcm.predict(x)

f, axes = plt.subplots(1, 2)

# Plotting Raw data on Plot 1
axes[0].scatter(x[:, 0], x[:, 1])

# Plotting Clusters on Plot 2 with different colors
axes[1].scatter(x[:, 0], x[:, 1], c=fcm_labels)
axes[1].scatter(fcm_centers[:, 0], fcm_centers[:, 1], marker="+", c="red")
plt.show()

