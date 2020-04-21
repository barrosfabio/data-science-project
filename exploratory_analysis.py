# Import all packages you need here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from statistics import stdev, mode

# load your data here
data_frame = pd.read_csv('Fma_sl_ssd_full.csv')
rows, cols = data_frame.shape
print('Old DataFrame shape - rows: {} cols:{}'.format(rows, cols))

# For now, let's drop the missing data (rows with all 0s)
data_frame = data_frame[(data_frame.sum(axis=1) != 0)]
rows, cols = data_frame.shape
print('New DataFrame shape - rows: {} cols:{}'.format(rows, cols))


# Setting up slices
feature_position = np.zeros(24)
j=0
for i in range(0,cols):
  if(i % 7 == 0):
    feature_position[j] = int(i)
    j+=1
print(feature_position)

# Let's filter the dataset by bark bands
bark_bands = np.arange(1,24,1)
print(bark_bands)

# Filtering only the columns for the mean of each bark critical band
data_frame_mean = data_frame.iloc[:, feature_position].copy()
rows, cols = data_frame_mean.shape
print('Filtered DataFrame shape - rows: {} cols:{}'.format(rows, cols))
#print(data_frame_mean.head(5))

# Filtering by genre
genre_list = ['Pop','Rock','Classical','Jazz']
pop = data_frame_mean['class'].str.contains('pop', case=False)
genre_filtered_data_frame = pd.DataFrame()

print('\nFiltering by genre')
for genre in genre_list:
  data = data_frame_mean[data_frame_mean['class'].str.startswith(genre)].copy()
  ## Re-label classes to its root class
  data['class'] = genre
  genre_filtered_data_frame = genre_filtered_data_frame.append(data)

rows, cols = genre_filtered_data_frame.shape
print('Filtered DataFrame shape - rows: {} cols:{}'.format(rows, cols))
print(genre_filtered_data_frame.head(10))


plt.rcParams['figure.figsize'] = (25,25)

# Let's try to do a correlation analysis with the 23 features:
correlation = genre_filtered_data_frame.corr()
sns.heatmap(correlation, annot=True, fmt='g')
plt.show()

#Plotting some graphs
# Plotting a Scatterplot of all shooting zones
sns.scatterplot(x=genre_filtered_data_frame['feature1'], y=genre_filtered_data_frame['feature148'], marker='o', hue=genre_filtered_data_frame['class'])
plt.xlabel('feature 1')
plt.ylabel('feature 155')
plt.title('Music Genres', loc='center')
plt.show()