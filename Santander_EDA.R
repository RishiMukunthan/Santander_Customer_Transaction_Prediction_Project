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

##################################Missing Values Analysis###############################################
#Create data frame with missing value count in each column
missing_val_train = data.frame(apply(data_train,2,function(x){sum(is.na(x))}))

missing_val_test = data.frame(apply(data_train,2,function(x){sum(is.na(x))}))

#Removing unneeded objects as R uses RAM.
rm(missing_val_train)
rm(missing_val_test)

pdf("Santander_EDA_R.pdf")
##################################Outlier Analysis###############################################
### BoxPlots
#Show boxplots for first few variables.
boxplotshow = function(data){
  for(i in 2:ncol(data)){
    boxplot(data[i],main = colnames(data[i]))
  }
}
par(mfrow=c(2, 4))    # set the plotting area into a 1*2 array
boxplotshow(data_train[,3:11])

##################################Distribution Analysis###############################################
library(funModeling) 
library(tidyverse) 
library(Hmisc)

plot_num(data_train[,3:12])

##################################Target Class Imbalance###############################################
table(data_train$target) #Use table function to print unique value and their count

dev.off()