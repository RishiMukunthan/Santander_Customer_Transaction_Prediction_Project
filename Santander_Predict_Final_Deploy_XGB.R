rm(list=ls(all=T))

setwd("C:/Users/RISHI MUKUNTHAN/Desktop/Data Science/Projects/Edwisor_Santander_Customer_Transaction")

getwd()

#Load Libraries with help of lapply function
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

lapply(x, require, character.only = TRUE)
rm(x)

#Load data into data object(data frame)
data_train = read.csv("train.csv", header = T, na.strings = c(" ", "", "NA"))
data_test = read.csv("test.csv", header = T, na.strings = c(" ", "", "NA"))

library(dplyr)
data_train = ovun.sample(target ~ ., data = data_train[-1], 
                                 method = "both", p=0.143,N=175000, seed = 42)$data



# install.packages('xgboost')
library(xgboost)
dtrain = xgb.DMatrix(data=as.matrix(data_train[,2:201]), 
                     label = as.matrix(data_train[,1]))

params = list(eta=0.2, max_depth=20, min_child_weight=0.5,
              lambda=100, gamma =1)

classifier_XGB = xgb.train(params=params,data =dtrain, nrounds = 200,
                           objective="binary:logistic", eval_metric='auc')

# Predicting the Test set results
prob_pred = predict(classifier_XGB,newdata = as.matrix(data_test[-1]))
y_pred = ifelse(prob_pred > 0.5, 1, 0)

data_test_predicted = cbind.data.frame(ID_code=data_test[,1],probability=prob_pred,
                                       target=y_pred)

# Writing a csv (output)
write.csv(data_test_predicted, "XGB_Predictions_R.csv", row.names = F)
