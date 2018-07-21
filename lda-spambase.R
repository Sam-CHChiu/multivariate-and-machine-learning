rm(list=ls())

## Load required libraries
library(data.table)
library(dplyr)
library(MASS)
library(caret)
library(ggplot2)
library(plotROC)

## Load in spam mail dataset
set.seed(1234)
setwd('~/projects/multivariate-and-machine-learning/')
spamtest = fread('spambase.txt')
setDT(spamtest)
dim(spamtest)

## Check correlation
cor = cor(spamtest[, sapply(spamtest, is.numeric), with = F], method = 'pearson')
cor.column = which(cor > 0.99999, arr.ind = T)[which(cor > 0.99999, arr.ind = T)[,1]!=which(cor > 0.99999, arr.ind = T)[,2],]
spamtest[,rownames(cor.column):=NULL]

## Create training and testing set
split = 0.7
index = sample(nrow(spamtest), round(nrow(spamtest)*split))
spam.train = spamtest[index]
spam.test  = spamtest[-index]


## descriptive statistics
spam.train %>% summary
ggplot(spam.train, aes(x = class, fill=class)) + 
  geom_bar() + 
  stat_count(aes(label=..count..), geom="text", vjust=-1, size=5) +
  labs(title='Frequency of Emails vs. Spams in Training Set', y='Frequency', x='Class') + ylim(c(0, 3200))

## Linear discriminative analysis
## 10-fold cv
ctrl = trainControl(method="cv", summaryFunction=twoClassSummary, classProbs=T, savePredictions = T)
lda.fit = train(class ~ ., data = spam.train, method = "lda", trControl = ctrl)
summary(lda.fit)

## prediciton
lda.pred = predict(lda.fit, spam.test)
actual.value = factor(spam.test$class)
confusionMatrix(lda.pred, actual.value, positive = 'spam')

confusionMatrix(lda.pred, actual.value, positive = 'spam')$overall[1] # accuracy
confusionMatrix(lda.pred, actual.value, positive = 'spam')$byClass[1:2] # sensitivity (recall), specificity

#ROC Curve and AUC

roc.plot = ggplot(lda.fit$pred, aes(m = spam, d = obs )) + 
  geom_roc(hjust = -0.4, vjust = 1.5) + coord_equal() + 
  xlab('1 - Specificity') + ylab('Sensitivity') + 
  labs(title='ROC Curve') + theme(plot.title = element_text(hjust = 0.5))
roc.plot
calc_auc(roc.plot)

  