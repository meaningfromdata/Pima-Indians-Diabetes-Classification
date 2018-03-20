# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:43:16 2018

@author: David

parts of code adapted from code here:
    http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
    https://www.kaggle.com/dbsnail/diabetes-prediction-over-0-86-accuracy
    https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
    
class imbalance adapted partly from:
    https://elitedatascience.com/imbalanced-classes


"""


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
di_df = pd.read_csv('C:\\Users\\David\\Documents\\Data Science Related\\Datasets\\pima-indians-diabetes\\diabetes.csv')




### checking class balance (levels of Outcome variable)
di_df.Outcome.value_counts()



''' MAYBE COUNT ZERO VALUES HERE (ARE THESE LEGIT DATA VALUES?)
### Check how many columns contain missing data
print(di_df.isnull().any().sum(), ' / ', len(di_df.columns))

### Check how many entries in total are missing 
print(di_df.isnull().any(axis=1).sum(), ' / ', len(di_df))
'''






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







### import sklearn functions for Logistic Regression Classifier and train/test split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

### split data into features and target
X = df_combined_upsamp.iloc[:,:-1]
y = df_combined_upsamp.iloc[:, -1]

### split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)


### check dimensions of train/test data
print(X_train.shape)
print(X_test.shape)
print(y_train.size)
print(y_test.size)


### instantiate and fit logistic regression model
logReg = LogisticRegression()
logReg.fit(X_train, y_train)


### predict on test set and assess accuracy
y_pred = logReg.predict(X_test)
print('Accuracy of logReg on test set: {:.2f}'.format(logReg.score(X_test, y_test)))


### import libraries for cross-validation 
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# from sklearn.metrics import accuracy_score
# from sklearn import datasets
# from sklearn.feature_selection import RFE # recursive feature elimination

