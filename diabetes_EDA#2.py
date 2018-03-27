# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:43:16 2018

@author: David

parts of code adapted from code here:
    http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
    https://www.kaggle.com/dbsnail/diabetes-prediction-over-0-86-accuracy
    



"""


c=pd.read_csv(StringIO(s))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.stats import spearmanr, pearsonr
plt.style.use('ggplot')


'''
import warnings
warnings.filterwarnings('ignore')
'''


### read CSV file containing BR census data 
### data from https://github.com/meaningfromdata/Pima-Indians-Diabetes-Classification/blob/master/diabetes.csv
di_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\pima-indians-diabetes\\diabetes.csv')
# di_df = pd.read_table('https://github.com/meaningfromdata/Pima-Indians-Diabetes-Classification/blob/master/diabetes.csv')
# di_df = pd.read_csv('https://github.com/meaningfromdata/Pima-Indians-Diabetes-Classification/blob/master/diabetes.csv', sep=',', header=[0], index_col=0)
# di_df = pd.read_csv('https://raw.githubusercontent.com/meaningfromdata/Pima-Indians-Diabetes-Classification/blob/master/diabetes.csv')
# di_df = pd.read_csv('https://raw.github.com/meaningfromdata/Pima-Indians-Diabetes-Classification/blob/master/diabetes.csv')
# di_df = pd.read_csv('https://rawgit.com/meaningfromdata/Pima-Indians-Diabetes-Classification/blob/master/diabetes_fromPandas.csv')
# https://github.com/meaningfromdata/Pima-Indians-Diabetes-Classification/blob/master/diabetes_fromPandas.csv

'''
### This doesn't work either for importing this csv from github
from io import StringIO
import requests
url='https://github.com/meaningfromdata/Pima-Indians-Diabetes-Classification/blob/master/diabetes_fromPandas.csv'
s=requests.get(url).text
di_df=pd.read_csv(StringIO(s))
'''


di_df.to_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\pima-indians-diabetes\\diabetes_fromPandas.csv', sep=',', index=False)


test_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\pima-indians-diabetes\\diabetes_fromPandas.csv')



### Check how many columns contain missing data
print(di_df.isnull().any().sum(), ' / ', len(di_df.columns))

### Check how many entries in total are missing 
print(di_df.isnull().any(axis=1).sum(), ' / ', len(di_df))


### see how many 0's there are in each column.  Some of these 0's entries need to be imputed  
print(di_df.apply(pd.value_counts).head())


### glimpse summary statistics for each column 
print(di_df.describe())


### checking class balance (levels of Outcome variable)
print(di_df.Outcome.value_counts())








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
# def impute_mean_byCol(df, field, val):
#    df[field] = df[field].replace(val, df[field].mean())



### compute z-scores by group for each column to look for outliers
di_df_zscores = di_df.groupby('Outcome').transform(lambda x: (x - x.mean()) / x.std())

### observation at index=579 has an obvious outlier (likely bad data) in SkinThickness
print(di_df.iloc[579])


### drop observation (row) with outlier for skin thickness
di_df = di_df.drop(di_df.index[579])


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



### another check on imputation results
print(di_df_imputed.apply(pd.value_counts).head())




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



