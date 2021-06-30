library(readr)
library(caret)
library(dplyr)
library(randomForest)

#datFile = file.path(getwd(), 'data', 'traindata.tsv')
#testFile = file.path(getwd(), 'data', 'testdata.tsv')
rawFile = file.path(getwd(), 'data', 'mergedata.tsv')

#train.data = read_tsv(datFile) %>%
#  select(-c('Sample', 'OS', 'Death')) %>%
#  tidyr::drop_na()
#test.data = read_tsv(testFile) %>%
#  select(-c('Sample', 'OS', 'Death')) %>%
#  tidyr::drop_na()

# prepare dataset
raw <- read_tsv(rawFile)%>%
  select(-c('Sample', 'OS', 'Death')) %>%
  tidyr::drop_na()
indexGood <- which(raw$Outcome == 'good')
indexPoor <- which(raw$Outcome == 'poor')
indexTest <- c(indexGood[sample(1:length(indexGood), 50, replace = F)], 
               indexPoor[sample(1:length(indexPoor), 50, replace = F)]) # half good half poor for test

test.data = raw[indexTest,]
train.data = raw[-indexTest,]
# for outcome, OS<24 and Status==1 (1:death, 0:alive) would be a poor outcome(=poor)
#              OS>=24 would be a good outcome(=good)
#              otherwise are wiped out.

# reference for svm part: 
# http://www.sthda.com/english/articles/36-classification-methods-essentials/144-svm-model-support-vector-machine-essentials/

set.seed(42)

# SVM
model.svm.linear <- train(
  Outcome ~., data = train.data, method = 'svmLinear',
  na.action = na.pass,
  tuneGrid = expand.grid(C = seq(0, 2, length = 20)),
  trControl = trainControl('cv', number = 10),
  preProcess = c('center', 'scale'),
  metric = 'Accuracy'
)
print(model.svm.linear)
# predict with test data set
predicted.svm.linear <- model.svm.linear %>% predict(test.data)
# accuracy
mean(predicted.svm.linear == test.data$Outcome)
# check the tuned cost
plot(model.svm.linear)
# seems it doesn't matter

# try non-linear kernel
model.svm.nonlinear <- train(
  Outcome ~., data = train.data, method = "svmRadial",
  na.action = na.pass,
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale"),
  tuneLength = 10,
  metric = 'Accuracy'
)
print(model.svm.nonlinear)
# predict with test data set
predicted.svm.nonlinear <- model.svm.nonlinear %>% predict(test.data)
# accuracy
mean(predicted.svm.nonlinear == test.data$Outcome)

# random forest
# default
model.rf.default <- train(
  Outcome ~., data = train.data, method = "rf",
  na.action = na.pass,
  trControl = trainControl("cv", number = 10),
  #preProcess = c("center","scale"),
  tuneGrid = expand.grid(.mtry=sqrt(ncol(train.data)-1)),
  metric = 'Accuracy'
)
print(model.rf.default)
predicted.rf.default <- model.rf.default %>% predict(test.data)
mean(predicted.rf.default == test.data$Outcome)

# try grid search for tuning parameter
model.rf.grid <- train(
  Outcome ~., data = train.data, method = "rf",
  na.action = na.pass,
  trControl = trainControl("cv", number = 10, search = 'grid'),
  preProcess = c("center","scale"),
  tuneGrid = expand.grid(.mtry= (1:15)),
  metric = 'Accuracy'
)
print(model.rf.grid)
predicted.rf.grid <- model.rf.grid %>% predict(test.data)
mean(predicted.rf.grid == test.data$Outcome)

