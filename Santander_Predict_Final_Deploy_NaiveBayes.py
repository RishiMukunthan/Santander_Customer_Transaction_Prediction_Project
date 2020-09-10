import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df_train[df_train.target==0]
df_minority = df_train[df_train.target==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=150000,
                                 random_state=42) # reproducible results

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

X = df_downsampled.iloc[:, 2:].values
y = df_downsampled.iloc[:, 1].values

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy={1:25000})
X_res, y_res = ros.fit_sample(X, y)

from sklearn.preprocessing import StandardScaler
mm=StandardScaler()
X_scaled=mm.fit_transform(X_res)

import joblib
joblib.dump(mm,'scaler_final.pkl')

#Under a minutes
# Training the Naive Bayes Classification model on the Training set
from sklearn.naive_bayes import GaussianNB
NBclassifier = GaussianNB()
NBclassifier.fit(X_scaled, y_res)

import joblib
joblib.dump(NBclassifier,'NB_Model_Final.pkl')

model = joblib.load('NB_Model_Final.pkl')
scaler = joblib.load('scaler_final.pkl')

X_test = df_test.iloc[:, 1:].values

X_test = scaler.transform(X_test)

y_score = model.predict_proba(X_test)
y_pred = (y_score[:,1] >= 0.5).astype(int)
df_test_submit = df_test['ID_code']
df_test_submit = pd.concat([df_test_submit,pd.Series(y_pred),pd.Series(y_score[:,1])], axis=1)
df_test_submit.columns =['ID_code','Target','Probability']
df_test_submit.to_csv('Test_Prediction_NB.csv', index=False)