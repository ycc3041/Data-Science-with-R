save.image(file="new.RData") 
load('C:/Users/Annie Chen/Desktop/IE 590/Final Exam/0418.RData')


# ---------- Import, subset, remove columns and rows that have too many NA
# import dataset
df0 <- read.csv('C:/Users/Annie Chen/Desktop/IE 590/Final Exam/2012_public_use_data_aug2016.csv')
dim(df0) # 6720*1119

# select midwest
df0 = df0[df0$REGION==2, ]
dim(df0) # 1459 * 1119

# select potential predictors
df0 = df0[, c(3, 1119, 4:445, 1049:1051, 1063:1070, 1095)]
dim(df0) # 1459 * 456

# remove columns that contain too many NA values (30%)
missing.ratio = apply(df0, 2, function(x) sum(is.na(x))/dim(df0)[1])
df1 = df0[, missing.ratio <= 0.3]
dim(df1) # 1459 * 189

# remove rows that contain too many NA values (50%)
# calculate number of NA for each row
count.na = rowSums(is.na(df1))
# calculate how many rows contain more than 50% of NAs, in this case is 95
num = length(count.na[count.na >= 95])
num
# find row indice for those contain more than 50% of NAs
r = order(count.na, decreasing = T)[1:num]
r
# delete rows which contain more than 50% of NAs
df1 = df1[-r, ]
dim(df1) # 1421 * 189

# ---------- Explore response variable
summary(df1$ELCLBTU)
hist(df1$ELCLBTU)
rsummary(log(df1$ELCLBTU+1))
hist(log(df1$ELCLBTU+1))
df2 = df1
df2$ELCLBTU = log(df2$ELCLBTU+1)
dim(df2) # 1421 * 189

# ---------- Impute NA
# plot missing
library(DataExplorer)
plot_missing(df1[, 1:50])
plot_missing(df1[, 51:100])
plot_missing(df1[, 101:150])
plot_missing(df1[, 151:189])

# set correct data type
con.cols = c('SQFT', 'NFLOOR', 'FLCEILHT', 'YRCON', 'WKHRS', 'NWKER', 'HEATP', 'COOLP', 'PCTERMN', 'LAPTPN', 'PRNTRN', 'SERVERN', 'LTOHRP', 'LTNHRP', 'FLUORP', 'DAYLTP', 'HDD65', 'CDD65', 'ELCLBTU')
dis.cols = setdiff(colnames(df2), con.cols)
df2[dis.cols] = lapply(df2[dis.cols], factor)
str(df2)

# remove factors that contain only one level
df2$FREESTN = NULL
df2$ELUSED = NULL
df2$MFUSED = NULL
dim(df2) # 1421 * 186

# impute missing values
library(missForest)
set.seed(101)
df3 = df2
df3$ELCLBTU = NULL
imp.df = missForest(df3, maxiter = 10, ntree = 100)
df3 = as.data.frame(imp.df[1])
df3$ELCLBTU = df2$ELCLBTU
colnames(df3) = colnames(df2)
dim(df3) # 1421 * 189

# ---------- Correlation analysis
# correlation between continuous variables
library(caret)
library(DataExplorer)
df4 = df3
cor.mat = cor(df4[,con.cols])
high.cor = findCorrelation(cor.mat, cutoff = 0.7)
high.cor

plot_correlation(df4[,con.cols])

# delete PCTERMN(# of computers) and CDD65
df4$PCTERMN = NULL
df4$CDD65 = NULL
dim(df4) # 1421 * 184

# continuous predictors
new.con.cols = c('SQFT', 'NFLOOR', 'FLCEILHT', 'YRCON', 'WKHRS', 'NWKER', 'HEATP', 'COOLP', 'LAPTPN', 'PRNTRN', 'SERVERN', 'LTOHRP', 'LTNHRP', 'FLUORP', 'DAYLTP', 'HDD65', 'ELCLBTU')
length(new.con.cols) # 17

# ---------- Split dataset
# split dataset
set.seed(101)
r = sample(nrow(df4), size = .8*nrow(df4), replace = F)
train.df = df4[r,]
test.df <- df4[-r,]
dim(train.df) # 1136 * 184
dim(test.df) # 285 * 184


# ---------- Feature selection
# variable importance

# use only continuous
t1 = lm(ELCLBTU ~ SQFT+NFLOOR+FLCEILHT+YRCON+WKHRS+NWKER+HEATP+COOLP+LAPTPN+PRNTRN+SERVERN+LTOHRP+LTNHRP+FLUORP+DAYLTP+HDD65, data = train.df)
summary(t1) # 0.3386

# use all 183 predictors
t2 = lm(ELCLBTU~., data = train.df)
summary(t2) # 0.9644 

# rf to do feature selection
library(randomForest)
set.seed(101)
t3 = randomForest(ELCLBTU~., data = train.df, ntree = 1000, importance = T)

importance(t3)
varImp(t3)

num.var = sum(varImp(t3)>=5)
num.var

impVar.df = data.frame(varImp(t3))
impVar.df$Var = row.names(impVar.df)
most.imp = impVar.df[order(-impVar.df$Overall),][1:num.var,] 
most.imp$Var

train.sub = train.df[, most.imp$Var]
train.sub$ELCLBTU = train.df$ELCLBTU

t4 = lm(ELCLBTU~., data = train.sub)
summary(t4) # 0.9596 using 28 important variables

# ---------- Modle building process

# (1) linear regression models 
# use all variables
library(caret)
ctr = trainControl(method = 'cv', number = 10, search = 'random')

lm.cv1 = train(ELCLBTU~.,
           data = train.df,
           trControl = ctr,
           method = 'lm',
           seed = 101)
lm.cv1
lm.cv1$resample
lm.cv1$finalModel

# default lm
m1 = lm(ELCLBTU~., 
        data = train.sub)

# lm using cv
lm.cv2 = train(ELCLBTU~.,
           data = train.sub,
           trControl = ctr,
           method = 'lm',
           seed = 101)
lm.cv2
lm.cv2$resample
m2 = lm.cv2$finalModel

# (2) stepwise lm
m3 = stepAIC(m1, 
             direction = 'both',
             trace = F)

# (3) random forest
library(randomForest)
library(caret)
library(e1071)

# default rf
m4 = randomForest(ELCLBTU~., 
                  data = train.sub,
                  importance = T)
m4

# tune mtry from 8 to 11
ctr = trainControl(method = 'cv', number = 10, search = 'grid')
grid = expand.grid(.mtry=c(8:11))
tune.rf = train(ELCLBTU~., 
           data = train.sub, 
           method = 'rf', 
           metric = 'RMSE', 
           tuneGrid = grid,
           trControl = ctr, 
           importance = T,
           seed = 101)
tune.rf
best.mtry = tune.rf$bestTune$mtry 
best.mtry

# tune mtry from 12 to 16
grid = expand.grid(.mtry=c(12:16))
tune.rf2 = train(ELCLBTU~., 
                data = train.sub, 
                method = 'rf', 
                metric = 'RMSE', 
                tuneGrid = grid,
                trControl = ctr, 
                importance = T,
                seed = 101)

tune.rf2

# tune mtry from 17 to 28
grid = expand.grid(.mtry=c(17:28))
tune.rf3 = train(ELCLBTU~., 
                 data = train.sub, 
                 method = 'rf', 
                 metric = 'RMSE', 
                 tuneGrid = grid,
                 trControl = ctr, 
                 importance = T,
                 seed = 101)

tune.rf3

best.mtry = tune.rf3$bestTune$mtry
best.mtry

# build rf with best mtry
m5 = randomForest(ELCLBTU~.,
                  data = train.sub,
                  ntree = 1000,
                  mtry = best.mtry,
                  seed = 101)

summary(m5)
importance(m5)

# (4) GAM
# smooth only on important continuous variables
# SQFT, COOLP, NFLOOR, PRNTRN
library(gam)
m6 = gam(ELCLBTU~.,
         family = 'gaussian',
         data = train.sub)
summary(m6)

# (5) MARS using earth
library(earth)
m7 = earth(ELCLBTU~.,
           data = train.sub,
           pmethod = 'cv',
           nfold = 10)
m7


earth.tune = tune(earth, 
                  ELCLBTU~., 
                  data = train.sub,
                  ranges = list(nprune=c(5, 10, 15, 20), degree = c(1:3)))

earth.tune
plot(earth.tune)
earth.tune$best.parameters # nprune = 20, degree = 2
m8 = earth.tune$best.model
m8

# (5) BART
options(java.parameters = "-Xmx12g") 
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jdk-11.0.1') 
library(rJava)
library(bartMachine)

m9 = bartMachine(
  X = train.sub[,-29], 
  y = train.sub$ELCLBTU,
  num_trees = 500
)
m9

# tune BART
tune.bart = bartMachineCV(X = train.sub[,-29],
                    y = train.sub$ELCLBTU,
                    num_tree_cvs = c(50, 200),
                    k_cvs = c(2, 3, 5),
                    nu_q_cvs = list(c(3, 0.9), c(3, 0.99), c(10, 0.75)),
                    k_folds = 5)

tune.bart$cv_stats


m13 = bartMachine(
  X = train.sub[,-29], 
  y = train.sub$ELCLBTU,
  num_trees = 200,
  k = 5,
  nu = 10,
  q = 0.75
) 

m13

# (6) SVM
library(e1071)
m10 = svm(ELCLBTU~., 
          data = train.sub)
m10

# tune SVM

svm.tune = tune(svm, 
           ELCLBTU~., 
           data = train.sub,
           ranges = list(epsilon=seq(0,1,0.1), cost = 2^(2:9)))

plot(svm.tune)
svm.tune$best.parameters
m11 = svm.tune$best.model
m11

# ---------- Model evaluation
test.sub = test.df[, most.imp$Var]
test.sub$ELCLBTU = test.df$ELCLBTU
dim(test.sub) # 258 * 29

library(Metrics)

all.mod = list(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11)


# in-sample rmse
inRMSE = c()

i = 1
for(mod in all.mod){
  if(i == 11){
    p = predict(m9, new_data = train.sub[,-29])
  } else{
    p = predict(mod, new_data = train.sub)
  }
  a = train.sub$ELCLBTU
  err = rmse(a, p)
  inRMSE = c(inRMSE, err)
  i = i + 1
  }
inRMSE

# out-of-sample rmse
outRMSE = c()
i = 1
for(mod in all.mod){
  if(i == 9){
    p = predict(m9, new_data = test.sub[,-29])
  } else{
    p = predict(mod, new_data = test.sub)
  }
  a = test.sub$ELCLBTU
  err = rmse(a, p)
  outRMSE = c(outRMSE, err)
  i = i + 1
}
outRMSE

# in-sample mae
inMAE = c()
i = 1
for(mod in all.mod){
  if(i == 9){
    p = predict(m9, new_data = train.sub[,-29])
  } else{
    p = predict(mod, new_data = train.sub)
  }
  a = train.sub$ELCLBTU
  err = mae(a, p)
  inMAE = c(inMAE, err)
  i = i + 1
}
inMAE

# out-of-sample mae
outMAE = c()
i = 1
for(mod in all.mod){
  if(i == 9){
    p = predict(m9, new_data = test.sub[,-29])
  } else{
    p = predict(mod, new_data = test.sub)
  }
  a = test.sub$ELCLBTU
  err = mae(a, p)
  outMAE = c(outMAE, err)
  i = i + 1
}
outMAE


p = predict(m13, new_data = test.sub[,-29])
a = test.sub$ELCLBTU
rmse(a, p)

# null model

p = mean(train.sub$ELCLBTU)
rmse(train.sub$ELCLBTU, p)
rmse(test.sub$ELCLBTU, p)
mae(train.sub$ELCLBTU, p)
mae(test.sub$ELCLBTU, p)


# ---------- Final model

finalmodel = m13
finalmodel

var_sel = var_selection_by_permute(finalmodel)
var_sel
save.image(var_sel, 'varsel.jpg')

var_sel$important_vars_local_names
