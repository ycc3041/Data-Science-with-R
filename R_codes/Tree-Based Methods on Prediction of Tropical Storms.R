
# 1. Importing Data set

df = read.csv('./TCcount.csv')
dim(df)
colnames(df) = c('Year', 'Storms', 'Landfalling', 'Temp', 'ENSO', 'NAO')
head(df)

# 2. Splitting Data Set

set.seed(111)
rows = sample(nrow(df), size = .8*nrow(df), replace = F)
train.df = df[rows,]
test.df = df[-rows,]

# 3. Building Regression Tree

library('rpart')
tree = rpart(Storms~Temp+ENSO+NAO, data=train.df, method='anova')
summary(tree)
library('rpart.plot')
rpart.plot(tree)
printcp(tree)

# 4. Pruning Tree

tree.pruned = prune(tree, cp=tree$cptable[which.min(tree$cptable[,'xerror']),'CP'])
summary(tree.pruned)

rpart.plot(tree.pruned)

# 5. Comparing Training Errors and Test Errors

library(MLmetrics)
train.prediction.1 = predict(tree, train.df)
test.prediction.1 = predict(tree, newdata=test.df)
train.prediction.2 = predict(tree.pruned, train.df)
test.prediction.2 = predict(tree.pruned, newdata=test.df)

training.rmse = c(RMSE(train.df$Storms, train.prediction.1), RMSE(train.df$Storms, train.prediction.2))
test.rmse = c(RMSE(test.df$Storms, test.prediction.1), RMSE(test.df$Storms, test.prediction.2))
model = c('tree', 'pruned tree')
table.1 = data.frame(model, training.rmse, test.rmse)
table.1

# 6. Building Classification Tree

value = quantile(df$Storms, 0.8)
train.df$Status = ifelse(train.df$Storms >= value, 1, 0)
test.df$Status = ifelse(test.df$Storms >= value, 1, 0)

set.seed(121)
k = 10
train.df$Fold = sample(1:k, size=nrow(train.df), replace=T)
head(train.df)

train.acc = c()
val.acc = c()

for (i in 1:k){
  train = train.df[-which(train.df$Fold==i),]
  val = train.df[which(train.df$Fold==i),]
  tree = rpart(Status ~ Temp+ENSO+NAO, data=train, method='class')
  train.acc = c(train.acc, Accuracy(predict(tree, type='class'), train$Status))
  val.acc = c(val.acc, Accuracy(predict(tree, newdata=val, type='class'), val$Status))
}

itr = c(1:10)
acc.df = data.frame(itr, train.acc, val.acc)
acc.df

library(ggplot2)
ggplot(acc.df, aes(itr)) + geom_line(aes(y=train.acc, colour='Training accuracy'), size=1) + geom_line(aes(y=val.acc, colour='Validation accuracy'), size=1) + labs(title='Accuracy Plot', x='Iteration', y='Accuracy', color='Legend') + scale_x_continuous(breaks=1:10)

sum(train.acc)/length(train.acc)
sum(val.acc)/length(val.acc)

# 7. Building Random Forests

library(randomForest)
tree.rmse = c()
forest.rmse = c()

for (i in 1:k){
  train = train.df[-which(train.df$Fold==i),]
  val = train.df[which(train.df$Fold==i),]
  tree = rpart(Storms~Temp+ENSO+NAO, data=train)
  forest = randomForest(Storms~Temp+ENSO+NAO, data=train, ntree=500)
  tree.rmse = c(tree.rmse, RMSE(val$Storms, predict(tree, newdata=val)))
  forest.rmse = c(forest.rmse, RMSE(val$Storms, predict(forest, newdata=val)))
}

acc.df.2 = data.frame(itr, tree.rmse, forest.rmse)
acc.df.2

ggplot(acc.df.2, aes(itr)) + geom_line(aes(y=tree.rmse, colour='Single Tree RMSE'), size=1) + geom_line(aes(y=forest.rmse, colour='Random Forest RMSE'), size=1) + labs(title='RMSE Plot', x='Iteration', y='RMSE', color='Legend') + scale_x_continuous(breaks=1:10)

sum(tree.rmse)/length(tree.rmse)
sum(forest.rmse)/length(forest.rmse)

rforest = randomForest(Storms~Temp+ENSO+NAO, data=train.df)
# training RMSE
RMSE(train.df$Storms, predict(rforest))
# test RMSE
RMSE(test.df$Storms, predict(rforest, newdata=test.df))


