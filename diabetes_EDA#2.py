# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:43:16 2018

@author: David

parts of code adapted from code here:
    http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
    https://www.kaggle.com/dbsnail/diabetes-prediction-over-0-86-accuracy
    



"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
plt.style.use('ggplot')


'''
import warnings
warnings.filterwarnings('ignore')
'''



### read CSV file containing BR census data
di_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\pima-indians-diabetes\\diabetes.csv')


### Check how many columns contain missing data
print(di_df.isnull().any().sum(), ' / ', len(di_df.columns))

### Check how many entries in total are missing 
print(di_df.isnull().any(axis=1).sum(), ' / ', len(di_df))


### glimpse summary statistics for each column 
di_df.describe()


### checking class balance (levels of Outcome variable)
di_df.Outcome.value_counts()



''' MAYBE COUNT ZERO VALUES HERE (ARE THESE LEGIT DATA VALUES?)
### Check how many columns contain missing data
print(di_df.isnull().any().sum(), ' / ', len(di_df.columns))

### Check how many entries in total are missing 
print(di_df.isnull().any(axis=1).sum(), ' / ', len(di_df))
'''





### boxplots for each column comparing values of features grouped by outcome (0 or 1)
plt.figure()
for i in range(1, di_df.shape[1]):
    plt.subplot(3, 3, i)
    plt.subplots_adjust(hspace = 0.35) # pads the vertical space between subplots
    sns.boxplot(x='Outcome', y=di_df.columns[i-1], data=di_df, palette='Set3')






### violinplots for each column comparing values of features grouped by outcome (0 or 1)
plt.figure()
for i in range(1, di_df.shape[1]):
    plt.subplot(3, 3, i)
    plt.subplots_adjust(hspace = 0.35) # pads the vertical space between subplots
    sns.violinplot(x='Outcome', y=di_df.columns[i-1], data=di_df, palette='Set3')





### function to do replacement of a value in a column by simple mean imputation over the whole column 
def impute_mean_byCol(df, field, val):
    df[field] = df[field].replace(val, df[field].mean())




### function to replace particular value in a feature column with mean of subset
### of column grouped by target variable level (i.e. mean imputation by group)
def impute_mean_byGroup(df, field, val, target):
    df_copy = df.copy() # make copy of df (not strictly necessary?)
    df_copy[field].replace(val, np.nan, inplace=True) # replace values of field that match val with nan
    field_grouped = df_copy[field].groupby(df_copy[target])  # field grouped by target (creates group object)
    field_nanRepByMean = field_grouped.apply(lambda x: x.fillna(x.mean())) # replace nan in each group by mean of field for that group  
    df_copy[field] = field_nanRepByMean.values
    return(df_copy)

### impute mean for 0 values by grouped outcome for selected columns 
di_df_imputed = di_df.copy()
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    di_df_imputed = impute_mean_byGroup(di_df_imputed, col, 0, 'Outcome')


### check that min value in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] is no longer 0
di_df_imputed.describe()


### visual check of imputation via violinplots for each column 
plt.figure()
for i in range(1, di_df_imputed.shape[1]):
    plt.subplot(3, 3, i)
    plt.subplots_adjust(hspace = 0.35) # pads the vertical space between subplots
    sns.violinplot(x='Outcome', y=di_df_imputed.columns[i-1], data=di_df_imputed, palette='Set3')


### computing and visualizing the correlation matrix for features 
# di_df_imputed.iloc[:,:-1].corr()
sns.heatmap(di_df_imputed.iloc[:,:-1].corr())
plt.yticks(rotation=0) # orients the y-axis row labels horizontally
plt.show()










### create accumulator df for storing difference of means 
### of resampled features grouped by outcome
diff_df=pd.DataFrame(columns=di_df_imputed.columns)


### generate 10000 bootstrapped differences of means for each for each feature grouped by outcome 

grouped = di_df_imputed.groupby('Outcome')

for i in range(1,10000):
    sampled = grouped.apply(lambda x: x.sample(frac=1.0, replace=True))
    sampledMeans = sampled.groupby('Outcome').mean()
    diff_temp = sampledMeans.diff().iloc[1,:]  # difference taken as mean_outcome1 - mean_outcome0
    diff_df = diff_df.append(diff_temp, ignore_index=True)


### distplots for differences of bootstrapped means for each feature by group
plt.figure()
for i in range(1, diff_df.shape[1]):
    plt.subplot(3, 3, i)
    plt.subplots_adjust(hspace = 0.35) # pads the vertical space between subplots
    sns.distplot(diff_df.iloc[:,i-1])
    


### compute pval by computing proportion of differences of mean
### that are 0 or less in each column
for col in diff_df.columns:
    pval = (diff_df[col]<0).sum()/len(diff_df[col])
    print(col + ' p-val:' + str(pval))


### diff_df.describe confirms by inspection that no distributions 
### contain zero
diff_df.describe()






### split data into features and target

X = di_df_imputed.iloc[:,:-1]
y = di_df_imputed.iloc[:, -1]

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)
print(X_train.shape)
print(X_test.shape)
print(y_train.size)
print(y_test.size)



