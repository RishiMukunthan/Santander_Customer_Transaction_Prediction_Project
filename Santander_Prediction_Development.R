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

##################################Preprocessing###############################################

#Train Test Split
#install.packages('caTools')
library(caTools)

set.seed(123)
split = sample.split(data_train$target, SplitRatio = 0.8)
#Press F1 for func help. Only give Y not X. Different from python.
#True if observation is decided to training set false then rest 0.2 percent.
training_set = subset(data_train[,2:202], split == TRUE)
test_set = subset(data_train[,2:202], split == FALSE)

# Feature Scaling - Normalization

#Testing set
for(i in colnames(test_set)){
  test_set[,i] = (test_set[,i] - min(training_set[,i]))/
    (max(training_set[,i] - min(training_set[,i])))
}

#Training set
for(i in colnames(training_set)){
  training_set[,i] = (training_set[,i] - min(training_set[,i]))/
    (max(training_set[,i] - min(training_set[,i])))
}

# #Standardisation
# for(i in colnames(training_set)){
#   training_set[,i] = (training_set[,i] - mean(training_set[,i]))/
#                                  sd(training_set[,i])
# }

##################################Modelling###############################################
##################################LOGISTIC REGRESSION#####################################
#Fitting Logistic Regression to the Training set
#About - glm-generalized linear model, dot to take all features,
#for logistic regression family should be binomial
classifier_logit = glm(formula = target ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier_logit, type = 'response', newdata = test_set[-1])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set$target, y_pred)
print(cm)

#Accuracy
sum(diag(cm))/nrow(test_set)

rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per 

#Precision , Recall , F-Score for both class
precision = diag(cm) / colsums 
recall = diag(cm) / rowsums 
f1 = 2 * precision * recall / (precision + recall) 

clf_report = data.frame(precision, recall, f1) 
print(clf_report)

#ROC_AUC Score
#install.packages('cvAUC')
library(cvAUC)

auc = AUC(prob_pred, test_set$target)
print(auc)

library(pROC)

dev.off() #Remove plots
par(pty='s')
roc(test_set$target, prob_pred, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab='FPR(1-Specificity)', ylab='TPR(Sensitivity)')

##################################Decision Tree###############################################
#Develop Model on training data
install.packages('C50')
library(C50)
C50_model = C5.0(formula = target ~., data = training_set)

# Predicting the Test set results
prob_pred = predict(C50_model, type = 'prob', newdata = test_set[-1])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set$target, y_pred[,2])
print(cm)

#Accuracy
sum(diag(cm))/nrow(test_set)

rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per 

#Precision , Recall , F-Score for both class
precision = diag(cm) / colsums 
recall = diag(cm) / rowsums 
f1 = 2 * precision * recall / (precision + recall) 

clf_report = data.frame(precision, recall, f1) 
print(clf_report)

#ROC_AUC Score
library(cvAUC)
auc = AUC(prob_pred[,2], test_set$target)
print(auc)

dev.off() #Remove plots
par(pty='s')
roc(test_set$target, prob_pred[,2], plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab='FPR(1-Specificity)', ylab='TPR(Sensitivity)')

##################################Naive Bayes###############################################
#rm(classifier_logit)
# install.packages('e1071')
library(e1071)
classifier_NB = naiveBayes(x = training_set[-1],
                        y = training_set$target)

# Predicting the Test set results
prob_pred = predict(classifier_NB,newdata = test_set[-1],type='raw')
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set$target, y_pred[,2])
print(cm)

#Accuracy
sum(diag(cm))/nrow(test_set)

rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per 

#Precision , Recall , F-Score for both class
precision = diag(cm) / colsums 
recall = diag(cm) / rowsums 
f1 = 2 * precision * recall / (precision + recall) 

clf_report = data.frame(precision, recall, f1) 
print(clf_report)

#ROC_AUC Score

auc = AUC(prob_pred[,2], test_set$target)
print(auc)

dev.off() #Remove plots
par(pty='s')
roc(test_set$target, prob_pred[,2], plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab='FPR(1-Specificity)', ylab='TPR(Sensitivity)')

##################################Random Forest###############################################
rm(classifier_RF)
#install.packages('randomForest')
library(randomForest)
classifier_RF = randomForest(x = training_set[-1],
                          y = training_set$target,
                          ntree = 10)

# Predicting the Test set results
prob_pred = predict(classifier_RF,newdata = test_set[-1],type='prob')
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set$target, y_pred[,2])
print(cm)

#Accuracy
sum(diag(cm))/nrow(test_set)

rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per 

#Precision , Recall , F-Score for both class
precision = diag(cm) / colsums 
recall = diag(cm) / rowsums 
f1 = 2 * precision * recall / (precision + recall) 

clf_report = data.frame(precision, recall, f1) 
print(clf_report)

#ROC_AUC Score

auc = AUC(prob_pred[,2], test_set$target)
print(auc)

dev.off() #Remove plots
par(pty='s')
roc(test_set$target, prob_pred[,2], plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab='FPR(1-Specificity)', ylab='TPR(Sensitivity)')


##################################XGBoost###############################################
library(xgboost)
#rm(dtrain,dtest,classifier_XGB)
dtrain = xgb.DMatrix(data=as.matrix(training_set[,-1]), 
                     label = as.matrix(training_set[,1]))
dtest = xgb.DMatrix(data=as.matrix(test_set[,-1]), 
                    label = as.matrix(test_set$target))

params = list(eta=0.3, max_depth=16, min_child_weight=0.5,
              lambda=100, gamma =1)

classifier_XGB = xgb.train(params=params,data =dtrain, nrounds = 100,
                         objective="binary:logistic", eval_metric='auc',
                         watchlist=list(train=dtrain,test=dtest),
                         early_stopping_rounds=5)

# Predicting the Test set results
prob_pred = predict(classifier_XGB,newdata = as.matrix(test_set[-1]))
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set$target, y_pred)
print(cm)

#Accuracy
sum(diag(cm))/nrow(test_set)

rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per 

#Precision , Recall , F-Score for both class
precision = diag(cm) / colsums 
recall = diag(cm) / rowsums 
f1 = 2 * precision * recall / (precision + recall) 

clf_report = data.frame(precision, recall, f1) 
print(clf_report)

#ROC_AUC Score

auc = AUC(prob_pred, test_set$target)
print(auc)

dev.off() #Remove plots
par(pty='s')
roc(test_set$target, prob_pred, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab='FPR(1-Specificity)', ylab='TPR(Sensitivity)')

##################################Modelling With Sampling###############################################
#SMOTE
#data_train$target = as.factor(data_train$target)
#data_smoted = SMOTE(target ~ ., data = data_train[-1],k=5, perc.over =100,perc.under = 200)
#table(data_smoted$target)

#install.packages("ROSE")
#library(ROSE)
library(dplyr)
#data_balanced = ROSE(target ~ ., data = data_train[-1], seed = 1)$data

#table(data_balanced$target)
#table(data_train$target)

#data_under_rose = data_balanced[sample(nrow(data_balanced), size=25000, replace =F),]

data_balanced_both = ovun.sample(target ~ ., data = data_train[-1], 
                                 method = "both", p=0.143,N=175000, seed = 42)$data
table(data_balanced_both$target)


#data_balanced_final = bind_rows(data_balanced_both,data_under_rose)
#table(data_balanced_final$target)
library(caTools)

set.seed(123)
split = sample.split(data_balanced_both$target, SplitRatio = 0.8)
#Press F1 for func help. Only give Y not X. Different from python.
#True if observation is decided to training set false then rest 0.2 percent.
training_set = subset(data_balanced_both, split == TRUE)
test_set = subset(data_balanced_both, split == FALSE)

#Feature Scaling

#Standardisation
#Testing set
for(i in colnames(test_set[-1])){
  test_set[,i] = (test_set[,i] - mean(training_set[,i]))/
    sd(training_set[,i])
}
#Training set
for(i in colnames(training_set[-1])){
  training_set[,i] = (training_set[,i] - mean(training_set[,i]))/
                                 sd(training_set[,i])
}


##################################Naive Bayes###############################################
rm(data_balanced_both,training_set,test_set)
# install.packages('e1071')
library(e1071)
classifier_NB = naiveBayes(x = training_set[-1],
                           y = training_set$target)

# Predicting the Test set results
prob_pred = predict(classifier_NB,newdata = test_set[-1],type='raw')
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set$target, y_pred[,2])
print(cm)

#Accuracy
sum(diag(cm))/nrow(test_set)

rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per 

#Precision , Recall , F-Score for both class
precision = diag(cm) / colsums 
recall = diag(cm) / rowsums 
f1 = 2 * precision * recall / (precision + recall) 

clf_report = data.frame(precision, recall, f1) 
print(clf_report)

#ROC_AUC Score

auc = AUC(prob_pred[,2], test_set$target)
print(auc)

dev.off() #Remove plots
par(pty='s')
roc(test_set$target, prob_pred[,2], plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab='FPR(1-Specificity)', ylab='TPR(Sensitivity)')