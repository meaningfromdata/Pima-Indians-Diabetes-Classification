# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:43:16 2018

@author: David

parts of code adapted from code here:
    https://www.kaggle.com/dbsnail/diabetes-prediction-over-0-86-accuracy
    https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
    
class imbalance adapted partly from:
    https://elitedatascience.com/imbalanced-classes
    
ROC Curve plotting adapted from:
    https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/


"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


'''
import warnings
warnings.filterwarnings('ignore')
'''



### read CSV file containing BR census data
di_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\pima-indians-diabetes\\diabetes.csv')




### checking class balance (levels of Outcome variable)
di_df.Outcome.value_counts()




### Check how many columns contain missing data
print(di_df.isnull().any().sum(), ' / ', len(di_df.columns))

### Check how many entries in total are missing 
print(di_df.isnull().any(axis=1).sum(), ' / ', len(di_df))




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


### assess balance of classes (counts of 0 and 1)
di_df_imputed['Outcome'].value_counts()



### copy di_df_imputed to new df simply called 'df' to use for modifying entries for balancing classes
df = di_df_imputed.copy()




### using upsampling (resampling) of minority class (1) to address class imbalance
from sklearn.utils import resample


### Separate majority and minority classes into new dfs
df_maj = df[df.Outcome==0]
df_min = df[df.Outcome==1]
 

### Upsample minority class
df_min_upsamp = resample(df_min, 
                                 replace=True,     # sample with replacement
                                 n_samples=500,    # to match majority class
                                 random_state=123) # reproducible results
 

### Combine majority class with upsampled minority class
df_combined_upsamp = pd.concat([df_maj, df_min_upsamp])
 

### Display new class counts
df_combined_upsamp.Outcome.value_counts()



### split data into features and target
X = df_combined_upsamp.iloc[:,:-1]
y = df_combined_upsamp.iloc[:, -1]



### import sklearn functions for PCA, compute principal components 
### and display data in terms of principal components
from sklearn.decomposition import PCA 
pca = PCA(n_components=2)
pca.fit(X)
X_2D = pca.transform(X)

df_combined_upsamp['PCA1'] = X_2D[:, 0]
df_combined_upsamp['PCA2'] = X_2D[:, 1]

sns.lmplot('PCA1','PCA2', hue='Outcome', data=df_combined_upsamp, fit_reg=False)

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))




### import TSNE, compute principal components  
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_2D_tsne = tsne.fit_transform(X)

df_combined_upsamp['tSNE1'] = X_2D_tsne[:,0]
df_combined_upsamp['tSNE2'] =  X_2D_tsne[:,1]


sns.lmplot('tSNE1', 'tSNE2', hue='Outcome', data=df_combined_upsamp, fit_reg=False)
