# Made by Naman Garg
# Contact : <12112061@nitkkr.ac.in>

import pandas as pd

# The task did not require much cleaning,
# because already cleaned data was provided

# importing dataset
dataset = pd.read_csv("../music_data_source.csv")
# pd.set_option('display.max_columns', None)

# Attributes
attributes = list(dataset.columns)
# print(attributes)

print(dataset.describe())
# Now, we can clearly see that length is same for complete dataset
# So, we can remove length
dataset.drop(['length'], axis=1, inplace=True)
print(dataset.describe())

# Save it to CSV
dataset.to_csv("../music_data_source_cleaned.csv", index=False)
