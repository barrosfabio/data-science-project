import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_combinations(path, combinations):

    for i in range(1,(len(path)+1)):
        combination = path[0:i]
        str_join = "/".join(combination)
        combinations.append(str_join)

    return combinations

def get_possible_classes(classes):
    combinations = []

    for i in range(len(classes)):
        possible_class = str(classes[i])
        class_splitted = possible_class.split('/')
        combinations = create_combinations(class_splitted, combinations)

    combinations = np.unique(combinations)

    return combinations

def append_to_class():

    data_frame = pd.read_csv('Fma_sl_ssd_full.csv')

    for i in range(len(data_frame)):
        row = data_frame.iloc[i,:]
        data_class = row['class']
        data_class = ''.join(('R/', data_class))
        data_frame.iloc[i,-1] = data_class

    data_frame.to_csv('Fma_sl_ssd_full_edited.csv', index=False)

def rename_class(df):

    for i in range(len(df)):
        data_class = df.iloc[i, -1]
        class_splitted = data_class.split('/')
        root_class = class_splitted[0:2]
        str_join = '/'.join(root_class)
        df.iloc[i, -1] = str_join
        #print(str_join)

    return df
    #df.to_csv('Fma_sl_ssd_full_root_genres.csv', index=False)


data_frame = pd.read_csv('Fma_sl_ssd_full_root_genres.csv')
stats = data_frame.describe()
for row in stats.iterrows():
    print(row)
#classes = data_frame['class']
#data_frame = rename_class(data_frame)
#data_frame_rock = data_frame[data_frame['class'].str.contains('Rock')]
#mean = data_frame_rock.mean()
#plt.rcParams['figure.figsize'] = (15,15)
#sns.lineplot(x=mean)
#plt.show
