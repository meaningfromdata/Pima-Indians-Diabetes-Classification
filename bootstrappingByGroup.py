# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:14:05 2018

@author: David
"""

### group by outcome
grouped = di_df_imputed.groupby('Outcome')




for i in range(1,100,1):
    sampled = grouped.apply(lambda x: x.sample(frac=1.0, replace=True))
    sampled.mean()


grouped = di_df_imputed.groupby('Outcome')

### bootstrap resampling by group
sampled = grouped.apply(lambda x: x.sample(n=10, replace=True))
sampledMeans = sampled.groupby('Outcome').mean()
diff_temp = sampledMeans.diff().iloc[1,:]



diff_df=pd.DataFrame(columns=di_df_imputed.columns)



sampled = grouped.apply(lambda x: x.sample(n=10, replace=True))
sampledMeans = sampled.groupby('Outcome').mean()
diff_temp = sampledMeans.diff().iloc[1,:]
diff_df = diff_df.append(diff_temp, ignore_index=True)




grouped.min()


grouped.sample(frac=0.025, replace=True)