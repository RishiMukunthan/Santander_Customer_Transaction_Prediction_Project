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

#Under 60 minutes Original with 200 variables
# Training XGBoost on the Training set
from xgboost import XGBClassifier
XGB_classifier = XGBClassifier(n_estimators=200, subsample=1, learning_rate=0.2, gamma=1,
                               max_depth=20, min_child_weight=0.5, reg_alpha=0, reg_lambda=100,
                               colsample_bylevel=0.9, colsample_bytree=1, scale_pos_weight=6)
XGB_classifier.fit(X_res, y_res,eval_metric="auc", verbose=2)

import joblib
joblib.dump(XGB_classifier,'XGB_Model_Final.pkl')

model = joblib.load('XGB_Model_Final.pkl')

X_test = df_test.iloc[:, 1:].values


y_score = model.predict_proba(X_test)
y_pred = (y_score[:,1] >= 0.5).astype(int)
df_test_submit = df_test['ID_code']
df_test_submit = pd.concat([df_test_submit,pd.Series(y_pred),pd.Series(y_score[:,1])], axis=1)
df_test_submit.columns =['ID_code','Target','Probability']
df_test_submit.to_csv('Test_Prediction_XGB.csv', index=False)