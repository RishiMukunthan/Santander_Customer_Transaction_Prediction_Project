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

#Standardisation
#Testing set
for(i in colnames(data_test[2:201])){
  data_test[,i] = (data_test[,i] - mean(data_train[,i]))/
    sd(data_train[,i])
}
#Training set
for(i in colnames(data_train[3:202])){
  data_train[,i] = (data_train[,i] - mean(data_train[,i]))/
    sd(data_train[,i])
}

# install.packages('e1071')
library(e1071)
classifier_NB = naiveBayes(x = data_train[3:202],
                           y = data_train$target)

# Predicting the Test set results
prob_pred = predict(classifier_NB,newdata = data_test[-1],type='raw')
y_pred = ifelse(prob_pred > 0.5, 1, 0)

data_test_predicted = cbind.data.frame(ID_code=data_test[,1],probability=prob_pred[,2],
                                       target=y_pred[,2])

# Writing a csv (output)
write.csv(data_test_predicted, "NB_Predictions_R.csv", row.names = F)
