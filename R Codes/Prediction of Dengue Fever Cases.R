# Linear & GLMNET Modified
# Created: 4/6/19

# Load Libraries
library(readr)
library(glmnet)
library(ModelMetrics)
library(caret)
library(olsrr)
library(e1071)
library(imputeTS)
library(mgcv)
library(earth)
library(gbm)
library(randomForest)
library(e1071)
library(Metrics)




# Read Dataset
df0 = read.csv('dengue.csv')
dim(df0) # 1456 * 26
head(df0)
df <- df0

# Reorder Columns
# response var, city, yr, week, vegetation, temp, precipitation, humidity
# remove X, week_start_date
ordered.df = df0[,c('total_cases', 'city', 'year', 'weekofyear',
                    'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw',
                    'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
                    'reanalysis_dew_point_temp_k', 'station_avg_temp_c',
                    'reanalysis_max_air_temp_k', 'station_max_temp_c',
                    'reanalysis_min_air_temp_k', 'station_min_temp_c',
                    'reanalysis_tdtr_k', 'station_diur_temp_rng_c',
                    'precipitation_amt_mm', 'reanalysis_precip_amt_kg_per_m2',
                    'reanalysis_sat_precip_amt_mm', 'station_precip_mm',
                    'reanalysis_relative_humidity_percent',
                    'reanalysis_specific_humidity_g_per_kg'
)]
dim(ordered.df) # 1456*24
head(ordered.df)

# # EDA
# plot(x = df$weekofyear, y = df$total_cases, xlab = "Week of Year", ylab = "Total Dengue Cases in Week", main = "Annual Dengue Variation")
# plot(x = df$weekofyear, y = df$station_precip_mm, xlab = "Week of Year", ylab = "Average Precipitation (mm)", main = "Annual Precipitation Variation")
# plot(x = df$weekofyear, y = df$station_avg_temp_c, xlab = "Week of Year", ylab = "Average Temperature (C)", main = "Annual Temperature Variaton")
# plot(x = df$weekofyear, y = df$reanalysis_tdtr_k, xlab = "Week of Year", ylab = "Diurnal Temp Variaton (K)", main = "Annual Diurnal Temperature Variaton")
# plot(x = df$weekofyear, y = df$reanalysis_relative_humidity_percent, xlab = "Week of Year", ylab = "Relative Humidity (%)", main = "Annual Humidity Variaton")
# 
# # Divide into SJ and IQ for fun! 
# sj <- df[which(df$city=="sj"),]
# iq <- df[which(df$city=="iq"),]
# 
# # Comparative EDA
# par(mfrow = c(1, 2))
# plot(x = sj$weekofyear, y = sj$reanalysis_tdtr_k, xlab = "Week of Year", ylab = "Diurnal Temp Variaton (K)", main = "SJ Annual Diurnal Temperature Variaton", ylim = c(0,16))
# plot(x = iq$weekofyear, y = iq$reanalysis_tdtr_k, xlab = "Week of Year", ylab = "Diurnal Temp Variaton (K)", main = "IQ Annual Diurnal Temperature Variaton", ylim = c(0,16))
# 
# plot(x = sj$weekofyear, y = sj$total_cases, xlab = "Week of Year", ylab = "Total Dengue Cases in Week", main = "SJ Annual Dengue Variation", ylim = c(0,400))
# plot(x = iq$weekofyear, y = iq$total_cases, xlab = "Week of Year", ylab = "Total Dengue Cases in Week", main = "IQ Annual Dengue Variation", ylim = c(0,400))
# 
# plot(x = sj$weekofyear, y = sj$station_precip_mm, xlab = "Week of Year", ylab = "Average Precipitation (mm)", main = "SJ Annual Precipitation Variation", ylim = c(0,400))
# plot(x = iq$weekofyear, y = iq$station_precip_mm, xlab = "Week of Year", ylab = "Average Precipitation (mm)", main = "IQ Annual Precipitation Variation", ylim = c(0,400) )
# 
# plot(x = sj$weekofyear, y = sj$station_avg_temp_c, xlab = "Week of Year", ylab = "Average Temperature (C)", main = "SJ Annual Temperature Variaton", ylim = c(22,32))
# plot(x = iq$weekofyear, y = iq$station_avg_temp_c, xlab = "Week of Year", ylab = "Average Temperature (C)", main = "IQ Annual Temperature Variaton", ylim = c(22,32))
# 
# plot(x = sj$weekofyear, y = sj$reanalysis_relative_humidity_percent, xlab = "Week of Year", ylab = "Relative Humidity (%)", main = "SJ Annual Humidity Variaton", ylim = c(60,100))
# plot(x = iq$weekofyear, y = iq$reanalysis_relative_humidity_percent, xlab = "Week of Year", ylab = "Relative Humidity (%)", main = "IQ Annual Humidity Variaton", ylim = c(60,100))
# 
# plot(x = sj$weekofyear, y = sj$ndvi_nw, xlab = "Week of Year", ylab = "Vegetation", main = "SJ NW Vegetation", ylim = c(-.4, .4))
# plot(x = iq$weekofyear, y = iq$ndvi_nw, xlab = "Week of Year", ylab = "Vegetation", main = "IQ NW Vegetation", ylim = c(-.4, .4))
# 
# plot(x = sj$weekofyear, y = sj$ndvi_se, xlab = "Week of Year", ylab = "Vegetation", main = "SJ SE Vegetation", ylim = c(-.4, .4))
# plot(x = iq$weekofyear, y = iq$ndvi_se, xlab = "Week of Year", ylab = "Vegetation", main = "IQ SE Vegetation", ylim = c(-.4, .4))
# 
# plot(x = sj$weekofyear, y = sj$ndvi_ne, xlab = "Week of Year", ylab = "Vegetation", main = "SJ NE Vegetation", ylim = c(-.4, .4))
# plot(x = iq$weekofyear, y = iq$ndvi_ne, xlab = "Week of Year", ylab = "Vegetation", main = "IQ NE Vegetation", ylim = c(-.4, .4))
# 
# plot(x = sj$weekofyear, y = sj$ndvi_sw, xlab = "Week of Year", ylab = "Vegetation", main = "SJ SW Vegetation", ylim = c(-.4, .4))
# plot(x = iq$weekofyear, y = iq$ndvi_sw, xlab = "Week of Year", ylab = "Vegetation", main = "IQ SW Vegetation", ylim = c(-.4, .4))
# 

# Data Cleaning - Remove rows with too many NAs
# calculate number of NA for each row
count.na = rowSums(is.na(ordered.df))
# calculate how many rows contain more than 50% of NAs, in this case is 12
length(count.na[count.na>=12])
# find row indice for those contain more than 50% of NAs
r = order(count.na, decreasing = T)[1:10]
r
ordered.df[r,]
# delete rows which contain more than 50% of NAs
ordered.df = ordered.df[-r, ]
dim(ordered.df) # 1446*24

# Imputation Using Prior Value
ordered.df<-na.locf(ordered.df, option = "locf",na.remaining = "keep");
com.df <- ordered.df # complete dataframe after imputation

# Remove Year
com.df <- com.df[,-which(names(com.df)=="year")]

# Split Dataset
# 80% training, 20% test
set.seed(590)
r = sample(nrow(com.df), size = 0.8*nrow(com.df))
train.df = com.df[r,]
test.df = com.df[-r,]
dim(train.df)
dim(test.df)

# Set up In Sample Error DF
# Set up Best Model Comparison
in.sample <- data.frame(model = character(),
                         mae = numeric(),
                         rmse = numeric(), 
                         stringsAsFactors = FALSE)

# Set up Best Model Comparison
best.model <- data.frame(model = character(),
                         mae = numeric(),
                         rmse = numeric(), 
                         stringsAsFactors = FALSE)

# NULL MODEL ------------------------------------------------------------------------------------
# Calculate Null Model
null.df <- train.df[,1]
null.holdout <- test.df[,1]
null.model <- mean(null.df)

# Statistics for Null Model
null.test <- rep(null.model, times = length(null.holdout))
null.mae <- mae(actual = null.holdout, predicted = null.test)
null.rmse <- rmse(actual = null.holdout, predicted = null.test)

# In Sample Error
in.sample[1,"model"] <- "Null"
in.sample[1,"mae"] <- mae(actual = null.df, predicted = rep(null.model, times = length(null.df)))
in.sample[1,"rmse"] <- rmse(actual = null.df, predicted = rep(null.model, times = length(null.df)))

# Update Best Model Frame
best.model[1,"model"] <- "Null"
best.model[1,"mae"] <- null.mae
best.model[1,"rmse"] <- null.rmse


# LINEAR MODEL ------------------------------------------------------------------------------------
# Data Sets for Linear
linear.df <- train.df # Training dataset which will be used to build model
df_s <- linear.df[,-c(1:2)] # Remove total cases,  week of year
preprocessParams <- preProcess(df_s, method=c("center", "scale"))
transformed <- predict(preprocessParams, df_s)
linear.df <- cbind(linear.df[,c(1:2)], transformed)

linear.holdout <- test.df # Test dataset which will be used for Holdout comparison between models
df_s <- linear.holdout[,-c(1:2)] # Remove total cases,  week of year
preprocessParams <- preProcess(df_s, method=c("center", "scale"))
transformed <- predict(preprocessParams, df_s)
linear.holdout <- cbind(linear.holdout[,c(1:2)], transformed)

# Transform Total Cases
linear.df["total_cases"] <- log(1+linear.df[,1])
linear.holdout["total_cases"] <- log(1+linear.holdout[,1])

# Linear Default Model
lm.model <- lm(total_cases ~ .  , data = linear.df)
summary(lm.model)
par(mfrow = c(2, 2))
plot(lm.model)

# Variable Selection
ols_step_both_p(lm.model) 
# Identified Variables: reanalysis_tdtr_k + weekofyear + reanalysis_air_temp_k + ndvi_se + ndvi_ne + reanalysis_max_air_temp_k + station_diur_temp_rng_c

# Improve Model with Variable Selection
lm.model.mod <- lm(total_cases ~ reanalysis_tdtr_k + weekofyear + reanalysis_air_temp_k + ndvi_se + 
                     ndvi_ne + reanalysis_max_air_temp_k + station_diur_temp_rng_c, data = linear.df)

summary(lm.model.mod)
par(mfrow = c(2, 2))
plot(lm.model.mod)

# IN SAMPLE PERFORMANCE
# Test Model Performance
lm.test.trans <- predict(lm.model.mod, newdata=linear.df)

# Un-Transform
lm.test <- exp(lm.test.trans) - 1
lm.actual <- exp(linear.df$total_cases) - 1

# In Sample Update
in.sample[2,"model"] <- "Linear"
in.sample[2,"mae"] <- mae(actual = lm.actual, predicted = lm.test)
in.sample[2,"rmse"] <- rmse(actual = lm.actual, predicted = lm.test)

# OUT OF SAMPLE PERFORMANCE
# Test Model Performance
lm.test.trans <- predict(lm.model.mod, newdata=linear.holdout)

# Un-Transform
lm.test <- exp(lm.test.trans) - 1
lm.actual <- exp(linear.holdout$total_cases) - 1

# Best Model Update
best.model[2,"model"] <- "Linear"
best.model[2,"mae"] <- mae(actual = lm.actual, predicted = lm.test)
best.model[2,"rmse"] <- rmse(actual = lm.actual, predicted = lm.test)

# GLM MODEL ------------------------------------------------------------------------------------
# CV to find best GLMNET Model
# Parameters: Alpha

# Data Sets for GLMNET
glm.df <- train.df # Training dataset which will be used with CV
df_s <- glm.df[,-c(1:2)] # Remove total cases, week of year
preprocessParams <- preProcess(df_s, method=c("center", "scale"))
transformed <- predict(preprocessParams, df_s)
glm.df <- cbind(glm.df[,c(1:2)], transformed)

glm.holdout <- test.df# Test dataset which will be used for Holdout comparison between models
df_s <- glm.holdout[,-c(1:2)] # Remove total cases,  week of year
preprocessParams <- preProcess(df_s, method=c("center", "scale"))
transformed <- predict(preprocessParams, df_s)
glm.holdout <- cbind(glm.holdout[,c(1:2)], transformed)

# Transform Total Cases
glm.df["total_cases"] <- log(1+glm.df[,1])
glm.holdout["total_cases"] <- log(1+glm.holdout[,1])

# K-fold CV
k <- 10
set.seed(590)
glm.df$folds <- sample(x=1:k,size=nrow(glm.df),replace=T)

alpha<-seq(from = 0, to = 1, by = .01)
glm_comp <- data.frame(alpha = numeric(),
                       mae1 = numeric(), 
                       mae2 = numeric(), 
                       mae3 = numeric(), 
                       mae4 = numeric(), 
                       mae5 = numeric(), 
                       mae6 = numeric(), 
                       mae7 = numeric(), 
                       mae8 = numeric(),
                       mae9 = numeric(),
                       mae10 = numeric(),
                       mae_ave = numeric())

i_a <- 1
i_k <- 1
response_index <- which(names(glm.df)=="total_cases")

for (i_a in 1:length(alpha)) {
  i_k <- 1
  for (i in 1:k) {
    test_i <- which(glm.df$folds == i_k, arr.ind = TRUE)
    train.xy <- glm.df[-test_i,]
    test.xy <- glm.df[test_i,]
    y.train <- as.matrix(train.xy[,response_index]) 
    x.train <- as.matrix(train.xy[,-response_index])
    y.test <- as.matrix(test.xy[,response_index])
    x.test <- as.matrix(test.xy[,-response_index])
    
    glm.model <- glmnet(x = x.train, y = y.train, family = "gaussian", alpha = alpha[i_a])
    glm.test <- predict(glm.model, newx=x.test)[,dim(predict(glm.model, newx=x.test))[2]]
    glm_comp[i_a,1] <- alpha[i_a]
    glm_comp[i_a,1+i_k] <- mae(actual = y.test, predicted = glm.test)
    i_k <- i_k + 1
  }
  i_a <- i_a + 1
}

glm_comp$mae_ave <- rowMeans(glm_comp[,2:11], na.rm = FALSE)
best.alpha <- glm_comp[which.min(glm_comp$mae_ave),"alpha"]


# GLM Best Model
x.train <- data.matrix(glm.df[,-c(1,23)])
y.train <- data.matrix(glm.df[,1])
x.test <- data.matrix(glm.holdout[,-1])
y.test <- data.matrix(glm.holdout[,1])
glm.model <- glmnet(x = x.train, y = y.train, family = "gaussian", alpha = 0)
glm.test <- predict(glm.model, newx=x.test) [,dim(predict(glm.model, newx=x.test))[2]]
 
# Update In-Sample
in.sample[3,"model"] <- "GLM"
in.sample[3,"mae"] <- mae(actual = y.train, predicted = predict(glm.model, newx=x.train) [,dim(predict(glm.model, newx=x.train))[2]])
in.sample[3,"rmse"] <- rmse(actual = y.train, predicted = predict(glm.model, newx=x.train) [,dim(predict(glm.model, newx=x.train))[2]])

# Update Best Model (OUT OF SAMPLE)
best.model[3,"model"] <- "GLM"
best.model[3,"mae"] <- mae(actual = y.test, predicted = glm.test)
best.model[3,"rmse"] <- rmse(actual = y.test, predicted = glm.test)


# SVM MODEL ------------------------------------------------------------------------------------
# Data Sets for SVM
svm.df <- train.df # Training dataset which will be used with CV
df_s <- svm.df[,-c(1:2)] # Remove total cases, week of year
preprocessParams <- preProcess(df_s, method=c("center", "scale"))
transformed <- predict(preprocessParams, df_s)
svm.df <- cbind(svm.df[,c(1:2)], transformed)

svm.holdout <- test.df # Test dataset which will be used for Holdout comparison between models
df_s <- svm.holdout[,-c(1:2)] # Remove total cases, week of year
preprocessParams <- preProcess(df_s, method=c("center", "scale"))
transformed <- predict(preprocessParams, df_s)
svm.holdout <- cbind(svm.holdout[,c(1:2)], transformed)

# K-fold CV for cost 
k <- 10
set.seed(590)
svm.df$folds <- sample(x=1:k,size=nrow(svm.df),replace=T)

cost<-c(0.001, 0.01, 0.1, 1, 5, 10, 100)
svm_comp <- data.frame(cost = numeric(),
                       mae1 = numeric(), 
                       mae2 = numeric(), 
                       mae3 = numeric(), 
                       mae4 = numeric(), 
                       mae5 = numeric(), 
                       mae6 = numeric(), 
                       mae7 = numeric(), 
                       mae8 = numeric(),
                       mae9 = numeric(),
                       mae10 = numeric(),
                       mae_ave = numeric())

i_c <- 1
i_k <- 1

for (i_c in 1:length(cost)) {
  i_k <- 1
  for (i in 1:k) {
    test_i <- which(svm.df$folds == i_k, arr.ind = TRUE)
    train.xy <- svm.df[-test_i,]
    test.xy <- svm.df[test_i,]
    svm.model <- svm(total_cases ~ . - folds, data = train.xy, cost = cost[i_c])
    svm.test <- predict(svm.model, data = test.xy)
    svm_comp[i_c,1] <- cost[i_c]
    svm_comp[i_c,1+i_k] <- mae(actual = test.xy$total_cases, predicted = svm.test)
    i_k <- i_k + 1
  }
  i_c <- i_c + 1
}

svm_comp$mae_ave <- rowMeans(svm_comp[,2:11], na.rm = FALSE)
best.cost <- svm_comp[which.min(svm_comp$mae_ave),"cost"]

# SVM Best Model
svm.df <- svm.df[,-23] # remove "folds"
svm.model <- svm(total_cases ~., data = svm.df, cost = best.cost)
svm.test <- predict(svm.model, newdata = svm.holdout)


# In Sample Error
in.sample[4,"model"] <- "SVM"
in.sample[4,"mae"] <- mae(actual = svm.df$total_cases, predicted = predict(svm.model, newdata = svm.df))
in.sample[4,"rmse"] <- rmse(actual = svm.df$total_cases, predicted = predict(svm.model, newdata = svm.df))

# Update Best model - Out of Sample
best.model[4,"model"] <- "SVM"
best.model[4,"mae"] <- mae(actual = svm.holdout$total_cases, predicted = svm.test)
best.model[4,"rmse"] <- rmse(actual = svm.holdout$total_cases, predicted = svm.test)


# GAM MODEL -----------------------------------------------------------------------

gam0 <- gam( total_cases ~ city + s(weekofyear, bs = 'cc',k=52) + 
               s(ndvi_ne) + s(ndvi_nw) + s(ndvi_se) + s(ndvi_sw) + 
               s(reanalysis_air_temp_k) + s(reanalysis_avg_temp_k) + 
               s(station_avg_temp_c)+ s(reanalysis_min_air_temp_k) +
               s(reanalysis_max_air_temp_k)+ s(station_max_temp_c)+
               s(station_min_temp_c) +
               s(reanalysis_tdtr_k)+s(station_diur_temp_rng_c)+
               s(reanalysis_precip_amt_kg_per_m2) + 
               s(reanalysis_sat_precip_amt_mm)+
               s(station_precip_mm)+
               s(reanalysis_relative_humidity_percent)+
               s(reanalysis_specific_humidity_g_per_kg),
             family=negbin(3), data=linear.df,method="REML") #method="GCV"

summary(gam1)

par(mfrow = c(2,2))
plot(gam1)

par(mfrow = c(2,2))
gam.check(gam1)

GAM_MAE.train = mean(abs(linear.df$total_cases -predict(gam0,newdata = linear.df)));GAM_MAE.train

GAM_RMSE.train <- mean((linear.df$total_cases -predict(gam0,newdata = linear.df))^2)^0.5;GAM_RMSE.train

GAM_MAE = mean(abs(linear.holdout$total_cases -predict(gam0,newdata = linear.holdout)));GAM_MAE

GAM_RMSE <- mean((linear.holdout$total_cases -predict(gam0,newdata = linear.holdout))^2)^0.5;GAM_RMSE

# MARS MODEL -------------------------------------------------------------------
#there are two tuning parameters associated with MARS model: the degree of interactions and the number of retained terms.

MARS_p1 <- earth(total_cases ~.,data=linear.df,pmethod = "backward")
MARS_p1$selected.terms
summary(MARS_p1)
#summary(MARS_p) %>% .$coefficients %>% head(10)
plotmo(MARS_p1)

#adding interaction of terms
MARS_p2 <- earth(total_cases ~.,data=linear.df,pmethod = "backward",degree = 2)
summary(MARS_p2)
#summary(MARS_p) %>% .$coefficients %>% head(10)
plotmo(MARS_p1)

#gridsearch
hyper_grid_1 <- expand.grid(degree = 1:3, 
                            nprune = floor(seq(2, 40, length.out = 10)) )

# for reproducibiity
set.seed(590)

# cross validated model using caret package
tuned_mars1 <- train(
  x = subset(linear.df, select = -total_cases),
  y = linear.df$total_cases,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = hyper_grid_1
)

# best model
tuned_mars1$bestTune

#plot of tunning
RMSEcv_1 <- ggplot(tuned_mars1)+labs(x = "Number of retained terms")+theme(plot.title = element_text(face = "italic",hjust = 0.5))+ggtitle("Cross-validated RMSE for the 30 Different Hyperparameter Combinations");
RMSEcv_1

MARS_p3 <- earth(total_cases ~.,data=linear.df,pmethod = "backward",nprune=18,degree=1)
summary(MARS_p3)
#summary(MARS_p) %>% .$coefficients %>% head(10)
plotmo(MARS_p3)

# variable importance plots
p1 <- vip(MARS_p3, num_features = 34, bar = FALSE)+theme(plot.title = element_text(face = "italic",hjust = 0.5))+ggtitle("Variable Importance Based on Impact to GCV");p1

summary(MARS_p3)$coefficients
MARS_MAE.train = mean(abs(linear.df$total_cases -predict(MARS_p3,newdata = linear.df)));MARS_MAE.train
#16.85
MARS_RMSE.train <- mean((linear.df$total_cases -predict(MARS_p3,newdata = linear.df))^2)^0.5;MARS_RMSE.train
#27.29

MARS_MAE = mean(abs(linear.holdout$total_cases -predict(MARS_p3,newdata = linear.holdout)));MARS_MAE
#16.85
MARS_RMSE <- mean((linear.holdout$total_cases -predict(MARS_p3,newdata = linear.holdout))^2)^0.5;MARS_RMSE
#27.29

#RANDOM FOREST-------------------------------------------------------------------------------
customRF = list(type = 'Regression',
                library = 'randomForest',
                loop = NULL)

customRF$parameters = data.frame(parameter = c("mtry", "ntree"),
                                 class = rep("numeric", 2),
                                 label = c("mtry", "ntree"))

customRF$grid = function(x, y, len = NULL, search = "grid") {}

customRF$fit = function(x, y, wts, param, lev, last, weights, classProbs) {
  randomForest(x, y,
               mtry = param$mtry,
               ntree=param$ntree)
}

# predict label
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)

# predict prob
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")

customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

control <- trainControl(method='cv', 
                        number=10, 
                        allowParallel = TRUE)

tunegrid <- expand.grid(.mtry=c(7,9,11,13,15,17,19,21,23),.ntree=c(250,500,750))

set.seed(123)
custom <- train(total_cases~., 
                data=train.df, 
                method=customRF, 
                metric='RMSE', 
                tuneGrid=tunegrid, 
                trControl=control)

summary(custom)
custom$results
custom$bestTune
plot(custom)

# build rf with optimal hyperparameters
set.seed(590)
rf.m = randomForest(total_cases~., 
                    data = train.df,
                    mtry = 21,
                    ntree = 750,
                    importance = T
)

summary(rf.m)
plot(rf.m)plot

#BART -------------------------------------------------------------------------------
options(java.parameters = "-Xmx12g") 
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jdk-11.0.1') 
library(rJava)
library(bartMachine)

# default bart
bart.1 = bartMachine(
  X = train.df[,2:23], 
  y = train.df$total_cases,
  num_trees = 500
)

bart.1

# tune BART
tune.bart = bartMachineCV(X = train.df[,2:23],
                          y = train.df$total_cases,
                          num_tree_cvs = c(50, 200),
                          k_cvs = c(2, 3, 5),
                          nu_q_cvs = list(c(3, 0.9), c(3, 0.99), c(10, 0.75)),
                          k_folds = 5)

tune.bart$cv_stats

# build bart using optimal hyperparameters
set.seed(590)
bart.2 = bartMachine(
  X = train.df[,2:23], 
  y = train.df$total_cases,
  num_trees = 200,
  k = 3,
  nu = 3,
  q = 0.99
) 



#GBM-------------------------------------------------------------------------------

GB <- gbm(total_cases ~.,data=train.df, distribution="poisson", n.trees=50000, interaction.depth=2, shrinkage=0.1)

#Tunning
# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(5,7,9),
  n.minobsinnode = c(3,5,10),
  bag.fraction = c( 0.65,0.8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)

# randomize data
random_index <- sample(1:nrow(train.df), nrow(train.df))
random_train <- train.df[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(590)
  
  # train model
  gbm.tune <- gbm(
    formula = total_cases ~ .,
    distribution = "gaussian",
    data = random_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

#final model
set.seed(590)
# train the selected GBM model
GB <- gbm(
  formula = total_cases ~ .,
  distribution = "gaussian",
  data = train.df,
  n.trees = 1213,
  interaction.depth = 9,
  shrinkage = 0.01,
  n.minobsinnode = 3,
  bag.fraction = .8, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
  
)  
#variable importance
par(mar = c(5, 11, 1, 1))
summary(
  GB, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)
vp.plot<-vip::vip(GB);vp.plot

GB.pred <- predict(GB, n.trees = GB$n.trees, test.df)
GB_MAE = mean(abs(test.df$total_cases -GB.pred));GB_MAE
GB_RMSE <- mean((test.df$total_cases -GB.pred)^2)^0.5;GB_RMSE

GB.pred.train <- predict(GB, n.trees = GB$n.trees, train.df)
GB_MAE_train = mean(abs(train.df$total_cases -GB.pred.train));GB_MAE_train
GB_RMSE_train <- mean((train.df$total_cases -GB.pred.train)^2)^0.5;GB_RMSE_train



best.model[5,"model"] <- "GAM"
best.model[5,"mae"] <- 18.18207

best.model[6,"model"] <- "MARS"
best.model[6,"mae"] <- 13.935

best.model[7,"model"] <- "RF"
best.model[7,"mae"] <- 12.2434

best.model[8,"model"] <- "BART"
best.model[8,"mae"] <- 14.4095
