# problem 1
creative.df = read.csv("./creativeclass200711.csv", skip=0, stringsAsFactors = F)
education.df = read.csv("./Education.csv", skip=4, stringsAsFactors = F)
population.df = read.csv("./PopulationEstimates.csv", skip=2, stringsAsFactors = F)
poverty.df = read.csv("./PovertyEstimates.csv", skip=3, stringsAsFactors = F)
unemployment.df = read.csv("./Unemployment.csv", skip=7, stringsAsFactors = F)

calcuate_na = function(df){
  total = nrow(df) * ncol(df)
  ratio = sum(is.na(df)) / total
  return(ratio)
}

na.ratio = c()
df.row = c()
df.col = c()
for (df in list(creative.df, education.df, population.df, poverty.df, unemployment.df)){
  ratio = calcuate_na(df)
  na.ratio = c(na.ratio, ratio)
  df.row = c(df.row, nrow(df))
  df.col = c(df.col, ncol(df))
}

df.names = c('creative.df', 'education.df', 'population.df', 'poverty.df', 'unemployment.df')
ratio.df = data.frame(df.names, na.ratio, df.row, df.col)
print(ratio.df)


# merge dataframes
merged.df = merge(x=creative.df[, c(1:6, 8, 10)], y=education.df[, c(1, 40:47)], by.x = 'FIPS', by.y = 'FIPS.Code') 
merged.df = merge(x=merged.df, y=population.df[, c(1, 11:12, 27:28, 51:52, 59:60)], by.x = 'FIPS', by.y = 'FIPS') 
merged.df = merge(x=merged.df, y=poverty.df[, c(1, 8, 11, 14, 17, 26)], by.x = 'FIPS', by.y = 'FIPStxt') 
merged.df = merge(x=merged.df, y=unemployment.df[, c(1, 8:9, 12:13, 16:17, 20:21, 24:25)], by.x = 'FIPS', by.y = 'FIPStxt') 

new.df = merged.df
colnames(new.df) = c('FIPS', 'state', 'state.abr', 'country','metro',
                     'total.emp', 'total.creative','percent.creative', 'less.than.highschool','highschool.only',
                     'college', 'bachelor.higher', 'percent.less.than.highschool','percent.highschool.only', 'percent.college', 
                     'percent.bachelor.higher', 'pop.2010', 'pop.2011', 'birth.2010', 'birth.2011',
                     'international.mig.2010', 'international.mig.2011', 'domestic.mig.2010', 'domestic.mig.2011', 'all.pov.2016',
                     'percent.all.pov.2016', '0.17.pov.2016', 'percent.0.17.pov.2016', 'median.income.2016', 'employed.2007',
                     'unemployed.2007', 'employed.2008', 'unemployed.2008', 'employed.2009', 'unemployed.2009',
                     'employed.2010', 'unemployed.2010', 'employed.2011', 'unemployed.2011')

# problem 2
library(ggplot2)
ggplot(data = new.df, aes(x=total.creative)) + geom_density() + ggtitle('Density function of total creatives')
ggplot(data = new.df, aes(x=percent.creative)) + geom_density() + ggtitle('Density function of percent of creatives')


# problem 3
ggplot(new.df, aes(x=bachelor.higher, y=total.creative)) + geom_point(alpha = 0.3) + 
  ggtitle('Total creatives vs. number of bachelor or higher')

ggplot(new.df, aes(x=international.mig.2010, y=total.creative)) + geom_point(alpha = 0.3) +
  ggtitle('Total creatives vs. number of international immigration')

ggplot(new.df, aes(x=all.pov.2016, y=total.creative)) + geom_point(alpha = 0.3) +
  ggtitle('Total creatives vs. number of poverty')


# problem 4
new.df$metro = as.factor(new.df$metro)

# split data into training and test set
set.seed(111)
sample_row = sample(nrow(new.df), size = .8*nrow(new.df), replace = F)
train.df = new.df[sample_row,]
test.df = new.df[-sample_row,]

# simple regression models
simple.1 = lm(total.creative ~ total.emp, data = train.df)
simple.2 = lm(total.creative ~ bachelor.higher, data = train.df)
simple.3 = lm(total.creative ~ international.mig.2010, data = train.df)
simple.4 = lm(total.creative ~ all.pov.2016, data = train.df)
simple.5 = lm(total.creative ~ median.income.2016, data = train.df)

summary(simple.5)

library(Metrics)

i = 1
for (model in list(simple.1, simple.2, simple.3, simple.4, simple.5)){
  train.prediction = predict(model, train.df)
  test.prediction = predict(model, newdata = test.df)
  train.rmse = rmse(train.df$total.creative, train.prediction)
  test.rmse = rmse(test.df$total.creative, test.prediction)
  print(paste('# simple linear regression model', i))
  print(paste('RMSE on training set:', train.rmse))
  print(paste('RMSE on test set:', test.rmse))
  i = i+1
}

# multiple regression models
m1 = lm(total.creative ~. 
        -FIPS -state -state.abr -country -percent.creative
        -percent.less.than.highschool -percent.highschool.only -percent.college -percent.bachelor.higher - pop.2010 
        - pop.2011 -percent.all.pov.2016 -percent.0.17.pov.2016 -employed.2007 -employed.2008 
        -employed.2009 -employed.2010 -employed.2011, data = train.df)

summary(m1)

library(MASS)
m2.stepwise = stepAIC(m1, direction = 'both', trace = F)
summary(m2.stepwise)


i = 1
for (model in list(m1, m2.stepwise)){
  train.prediction = predict(m2.stepwise, train.df)
  test.prediction = predict(m2.stepwise, newdata = test.df)
  train.rmse = rmse(train.df$total.creative, train.prediction)
  test.rmse = rmse(test.df$total.creative, test.prediction)
  print(paste('# simple linear regression model', i))
  print(paste('RMSE on training set:', train.rmse))
  print(paste('RMSE on test set:', test.rmse))
  i = i+1
}

# problem 5
ex1 = new.df[which(new.df$country=='Los Alamos County'), ]
ex2 = new.df[which(new.df$country=='Fairfax County'), ]
ex3 = new.df[which(new.df$country=='Robeson County'), ]

prediction1 = predict(m1, ex1)
prediction2 = predict(m1, ex2)
prediction3 = predict(m1, ex3)

ex.df = data.frame(c('Los Alamos County', 'Fairfax County','Robeson County'),
                   c(prediction1, prediction2, prediction3),
                   c(ex1$total.creative, ex2$total.creative, ex3$total.creative))
colnames(ex.df) = c('County', 'prediction', 'true creatives')
ex.df

# problem 6
library(usmap)
map.df = new.df[,c('FIPS', 'percent.creative')]
colnames(map.df) = c('fips', 'value')
plot_usmap(regions = 'county', data=map.df, values='value') + 
  scale_fill_continuous(name = 'percent.creative', label = scales::comma) +
  theme(legend.position = 'right') + 
  labs(title = 'Percent of Creatives in US')
