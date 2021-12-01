library(caret)
library(missForest)
library(pROC)
library(ROCR)
library(DMwR)
library(FNN)
library(base)
library(doParallel)
setwd("C:/Users/wujin/Google ¶³ºÝµwºÐ/flu gitub")
training=read.csv("data_train.csv",header=TRUE,sep=",")
testing=read.csv("data_test.csv",header=TRUE,sep=",")

training$GeneXpert=ifelse(training$GeneXpert=="0","Negative","Positive")
training$GeneXpert=as.factor(training$GeneXpert)
table(training$GeneXpert)
colnames(training)

testing$GeneXpert=ifelse(testing$GeneXpert=="0","Negative","Positive")
testing$GeneXpert=as.factor(testing$GeneXpert)
table(testing$GeneXpert)
colnames(testing)

### Cross validation method: 5 numbers with 5 repeated
fitControl=trainControl(method="repeatedcv",
                        number=10,
                        repeats=10,
                        verbose = FALSE,
                        classProbs = TRUE,   
                        summaryFunction=twoClassSummary,
                        search="random")

### (1) CFOREST MODEL with tuneLength
set.seed(1234)
system.time(cforestmodelfit<-train(GeneXpert ~ .,
                                   data=training,
                                   method="cforest",
                                   metric="ROC",
                                   tuneLength=500,
                                   trControl=fitControl))

cforestmodelfit

predictions_train=predict(cforestmodelfit,newdata=training)
predictions_test=predict(cforestmodelfit,newdata=testing)
confusionMatrix(predict(cforestmodelfit,training),training$GeneXpert)
confusionMatrix(predict(cforestmodelfit,testing),testing$GeneXpert)

train_results=predict(cforestmodelfit,training,type="prob")
test_results=predict(cforestmodelfit,testing,type="prob")
train_results$obs=training$GeneXpert
train_results$pred=predictions_train
test_results$obs=testing$GeneXpert
test_results$pred=predictions_test
ROC_train<-roc(training$GeneXpert,train_results[,"Positive"],levels=c("Negative","Positive"))
ROC_test<-roc(testing$GeneXpert,test_results[,"Positive"],levels=c("Negative","Positive"))
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### ROC Curve for CFOREST MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")

### (2) Random Forest MODEL
system.time(rfmodelfit<-train(GeneXpert ~ .,
                              data=training,
                              method="rf",
                              metric="ROC",
                              tuneLength=500,
                              ntree = 200,
                              trControl=fitControl))

rfmodelfit

predictions_train=predict(rfmodelfit,newdata=training)
predictions_test=predict(rfmodelfit,newdata=testing)
confusionMatrix(predict(rfmodelfit,training),training$GeneXpert)
confusionMatrix(predict(rfmodelfit,testing),testing$GeneXpert)

train_results=predict(rfmodelfit,training,type="prob")
test_results=predict(rfmodelfit,testing,type="prob")
train_results$obs=training$GeneXpert
train_results$pred=predictions_train
test_results$obs=testing$GeneXpert
test_results$pred=predictions_test
ROC_train<-roc(training$GeneXpert,train_results[,"Positive"],levels=c("Negative","Positive"))
ROC_test<-roc(testing$GeneXpert,test_results[,"Positive"],levels=c("Negative","Positive"))
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### ROC Curve for Random Forest MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")

### (3) RANGER MODEL
rangermodelfit<-train(GeneXpert ~ .,
                      data=training,
                      method="ranger",
                      metric="ROC",
                      tuneLength=500,
                      trControl=fitControl)

rangermodelfit

predictions_train=predict(rangermodelfit,newdata=training)
predictions_test=predict(rangermodelfit,newdata=testing)
confusionMatrix(predict(rangermodelfit,training),training$GeneXpert)
confusionMatrix(predict(rangermodelfit,testing),testing$GeneXpert)

train_results=predict(rangermodelfit,training,type="prob")
test_results=predict(rangermodelfit,testing,type="prob")
train_results$obs=training$GeneXpert
train_results$pred=predictions_train
test_results$obs=testing$GeneXpert
test_results$pred=predictions_test
ROC_train<-roc(training$GeneXpert,train_results[,"Positive"],levels=c("Negative","Positive"))
ROC_test<-roc(testing$GeneXpert,test_results[,"Positive"],levels=c("Negative","Positive"))
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### ROC Curve for RANGER MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")

### (4) NNET MODEL
system.time(nnetmodelfit<-train(GeneXpert ~ .,
                                data=training,
                                method="nnet",
                                metric="ROC",
                                tuneLength=3000,
                                trControl=fitControl))

nnetmodelfit

predictions_train=predict(nnetmodelfit,newdata=training)
predictions_test=predict(nnetmodelfit,newdata=testing)
confusionMatrix(predict(nnetmodelfit,training),training$GeneXpert)
confusionMatrix(predict(nnetmodelfit,testing),testing$GeneXpert)

train_results=predict(nnetmodelfit,training,type="prob")
test_results=predict(nnetmodelfit,testing,type="prob")
train_results$obs=training$GeneXpert
train_results$pred=predictions_train
test_results$obs=testing$GeneXpert
test_results$pred=predictions_test
ROC_train<-roc(training$GeneXpert,train_results[,"Positive"],levels=c("Negative","Positive"))
ROC_test<-roc(testing$GeneXpert,test_results[,"Positive"],levels=c("Negative","Positive"))
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### ROC Curve for NNET MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")
vars<-varImp(nnetmodelfit)
vars

### (5) SVM MODEL
system.time(svmmodelfit<-train(GeneXpert ~ .,
                               data=training,
                               method="svmRadial",
                               metric="ROC",
                               tuneLength=500,
                               trControl=fitControl))

svmmodelfit

predictions_train=predict(svmmodelfit,newdata=training)
predictions_test=predict(svmmodelfit,newdata=testing)
confusionMatrix(predict(svmmodelfit,training),training$GeneXpert)
confusionMatrix(predict(svmmodelfit,testing),testing$GeneXpert)

train_results=predict(svmmodelfit,training,type="prob")
test_results=predict(svmmodelfit,testing,type="prob")
train_results$obs=training$GeneXpert
train_results$pred=predictions_train
test_results$obs=testing$GeneXpert
test_results$pred=predictions_test
ROC_train<-roc(training$GeneXpert,train_results[,"Positive"],levels=c("Negative","Positive"))
ROC_test<-roc(testing$GeneXpert,test_results[,"Positive"],levels=c("Negative","Positive"))
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### ROC Curve for SVM MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")
