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

# Renaming the columns / features to more meaningful names
genre_filtered_data_frame = genre_filtered_data_frame.rename(
  columns={'feature1': 'band_1', 'feature8': 'band_2', 'feature15': 'band_3', 'feature22': 'band_4',
           'feature29': 'band_5', 'feature36': 'band_6', 'feature43': 'band_7', 'feature50': 'band_8',
           'feature57': 'band_9', 'feature64': 'band_10', 'feature71': 'band_11', 'feature78': 'band_12',
           'feature85': 'band_13', 'feature92': 'band_14', 'feature99': 'band_15', 'feature106': 'band_16',
           'feature113': 'band_17', 'feature120': 'band_18', 'feature127': 'band_19', 'feature134': 'band_20',
           'feature141': 'band_21', 'feature148': 'band_22', 'feature155': 'band_23'}, errors='raise')


plt.rcParams['figure.figsize'] = (15,15)

# Filtering by genre
data_frame_rock = genre_filtered_data_frame.loc[genre_filtered_data_frame['class'] == 'Rock']
data_frame_pop = genre_filtered_data_frame.loc[genre_filtered_data_frame['class'] == 'Pop']
data_frame_classical = genre_filtered_data_frame.loc[genre_filtered_data_frame['class'] == 'Classical']
data_frame_jazz = genre_filtered_data_frame.loc[genre_filtered_data_frame['class'] == 'Jazz']

# Let's try to do a correlation analysis with the 23 features:
correlation = genre_filtered_data_frame.corr()
#sns.heatmap(correlation, annot=True, fmt='g')
#plt.show()

#Plotting some graphs
critical_band_1 = 'band_1'
critical_band_2 = 'band_22'

#Counting per class
sns.countplot(x='class', data=genre_filtered_data_frame)
plt.show()

## Univariate analysis

# Plotting distribution graphs
fig, axes = plt.subplots(3,1)
graph = sns.distplot(genre_filtered_data_frame[critical_band_1], kde=True, rug=False, ax=axes[0])
sns.distplot(genre_filtered_data_frame[critical_band_1], kde=False, rug=True, ax=axes[1])
sns.kdeplot(data=genre_filtered_data_frame[critical_band_1], ax=axes[2], shade=True)
plt.show()

# Plotting boxplots and violinplots
fig, axes = plt.subplots(3,1)
sns.boxplot(x = genre_filtered_data_frame[critical_band_1], ax=axes[0])
sns.boxenplot(x = genre_filtered_data_frame[critical_band_1], ax=axes[1])
sns.violinplot(x=genre_filtered_data_frame[critical_band_1], ax=axes[2])
plt.show()

# Plotting distribution graphs
fig, axes = plt.subplots(4,1)
axes[0].title.set_text("Rock")
axes[1].title.set_text("Pop")
axes[2].title.set_text("Classical")
axes[3].title.set_text("Jazz")
# sns.kdeplot(data=data_frame_rock[critical_band_1], ax=axes[0], shade=True, color='b')
# sns.kdeplot(data=data_frame_pop[critical_band_1], ax=axes[1], shade=True, color='r')
# sns.kdeplot(data=data_frame_classical[critical_band_1], ax=axes[2], shade=True, color='g')
# sns.kdeplot(data=data_frame_jazz[critical_band_1], ax=axes[3], shade=True, color='y')
sns.boxplot(x = data_frame_rock[critical_band_1], ax=axes[0])
sns.boxplot(x = data_frame_pop[critical_band_1], ax=axes[1])
sns.boxplot(x = data_frame_classical[critical_band_1], ax=axes[2])
sns.boxplot(x = data_frame_jazz[critical_band_1], ax=axes[3])
plt.show()

## Multivariate analysis
#Plotting a Scatterplot
sns.scatterplot(x=genre_filtered_data_frame[critical_band_1], y=genre_filtered_data_frame[critical_band_2], marker='o', hue=genre_filtered_data_frame['class'])
plt.xlabel(critical_band_1)
plt.ylabel(critical_band_2)
plt.title('Music Genres', loc='center')
plt.show()

# Plotting a scatterplot with transparency
sns.scatterplot(x=genre_filtered_data_frame[critical_band_1], y=genre_filtered_data_frame[critical_band_2], marker='o', hue=genre_filtered_data_frame['class'], alpha=0.3)
plt.xlabel(critical_band_1)
plt.ylabel(critical_band_2)
plt.title('Music Genres', loc='center')
plt.show()

# Plotting a KDE Plot
plt.show()
sns.kdeplot(data=genre_filtered_data_frame[critical_band_1], data2=genre_filtered_data_frame[critical_band_2], shade=True)
f = sns.FacetGrid(genre_filtered_data_frame, col='class', hue='class')
f = (f.map(sns.kdeplot, critical_band_1, critical_band_2))
plt.show()

# Plotting a Facet Grid
# Separating Shots Missed / Made from the right side
g = sns.FacetGrid(genre_filtered_data_frame, col='class', hue='class')
# Determine what kind of plot we want
g = (g.map(plt.scatter, critical_band_1, critical_band_2))
plt.show()

# Filtering Data by genre
genre_filtered_data_frame_1 = genre_filtered_data_frame[genre_filtered_data_frame['class'] == 'Rock']
# plot all the data first
sns.scatterplot(x=genre_filtered_data_frame[critical_band_1], y=genre_filtered_data_frame[critical_band_2], color='grey')
# plot the group we want to focus
sns.scatterplot(x=genre_filtered_data_frame_1[critical_band_1], y=genre_filtered_data_frame_1[critical_band_2], color='blue')
plt.show()

#Jointplot
g = sns.jointplot(x=genre_filtered_data_frame[critical_band_1], y=genre_filtered_data_frame[critical_band_2], kind='kde')
plt.show()

# Plotting a Joint Plot
with sns.axes_style("white"):
    sns.jointplot(x=genre_filtered_data_frame[critical_band_1], y=genre_filtered_data_frame[critical_band_2], kind="hex");
plt.show()
#Sampling
genre_filtered_data_frame_sample = genre_filtered_data_frame.sample(1000)
sns.scatterplot(x=genre_filtered_data_frame_sample[critical_band_1], y=genre_filtered_data_frame_sample[critical_band_2], marker='o', hue=genre_filtered_data_frame_sample['class'])
plt.show()

## Let's use continuous and categorical variables together
# Scatterplot
sns.scatterplot(x=genre_filtered_data_frame['class'], y=genre_filtered_data_frame[critical_band_1])
plt.show()

# Strip plot
sns.stripplot(x=genre_filtered_data_frame['class'], y=genre_filtered_data_frame[critical_band_1], size=5, jitter=0.2)
plt.show()

# Combination between strip plots and box-plots
sns.boxplot(x=genre_filtered_data_frame['class'], y=genre_filtered_data_frame[critical_band_1])
sns.stripplot(x=genre_filtered_data_frame['class'], y=genre_filtered_data_frame[critical_band_1], alpha=0.1, jitter=0.3)
plt.show()

# box-plots
sns.boxenplot(x=genre_filtered_data_frame['class'], y=genre_filtered_data_frame[critical_band_1])
plt.show()

# Violin plot with a strip plot behind
sns.violinplot(x=genre_filtered_data_frame['class'], y=genre_filtered_data_frame[critical_band_1])
sns.stripplot(x=genre_filtered_data_frame['class'], y=genre_filtered_data_frame[critical_band_1],alpha=0.3, jitter=0.3)
plt.show()

# Swarm plot
sns.swarmplot(x = genre_filtered_data_frame['class'], y = genre_filtered_data_frame[critical_band_1], size=4)
plt.show()

