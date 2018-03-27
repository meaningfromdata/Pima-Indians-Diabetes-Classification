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







### import sklearn functions for Quadratic Discriminant Analysis and train/test split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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


### instantiate and fit quadratic discriminant analysis 
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)


### predict on test set and assess accuracy
y_pred = qda.predict(X_test)
print('Accuracy of qda on test set: {:.2f}'.format(qda.score(X_test, y_test)))


### import libraries for cross-validation 
from sklearn import model_selection
from sklearn.model_selection import cross_val_score


### Perform 10-fold cross-validation and compute mean accuracy over folds
kfold = model_selection.KFold(n_splits=10)
modelCV = QuadraticDiscriminantAnalysis()
scoring = 'accuracy'
results = cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross-validation mean accuracy: %.3f" % (results.mean()))




### confusion matrix metrics and visualization
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

plt.figure(figsize=(9,9))
sns.heatmap(confusion_matrix, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
acc_title = 'Accuracy Score: {0}'.format(round(results.mean(),2))
plt.title(acc_title, size = 15);




### Compute ROC Curve and Plot
### calculate the fpr and tpr for all thresholds of the classification
from sklearn.metrics import roc_curve, auc

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


plt.title('ROC')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.025, 1.025])  # just giving a little padding to plot range
plt.ylim([-0.025, 1.025])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


