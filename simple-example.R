library(glmnet)
library(dplyr)
data("LifeCycleSavings")

LifeCycleSavings$work_pop <- 100 - LifeCycleSavings$pop15 - LifeCycleSavings$pop75
LifeCycleSavings <- select(LifeCycleSavings,
                           -c("pop15","pop75"))

train_rows <- sample(nrow(LifeCycleSavings), 40)
training_data <- LifeCycleSavings[train_rows, ]
testing_data <- LifeCycleSavings[-train_rows, ]

# make into matrices for glmnet compatibility
x.train <- as.matrix(training_data[,c("work_pop","dpi","ddpi")])
x.test <- as.matrix(testing_data[,c("work_pop","dpi","ddpi")])

y.train <- as.matrix(training_data[,c("sr")])
y.test <- as.matrix(testing_data[,c("sr")])

# train our models
linear <-lm(sr ~ ., data = training_data)

ridge <- glmnet::cv.glmnet(x = x.train, y = y.train, 
                           type.measure = "mse", family="gaussian",
                           alpha = 0, standardize = FALSE)

lasso <- glmnet::cv.glmnet(x = x.train, y = y.train, 
                           type.measure = "mse", family="gaussian",
                           alpha = 1, standardize = FALSE)

# Typically we don't explicitly set alpha for Elastic Net. 
# It would find the optimal alpha value on its own.
elasticNet <- glmnet::cv.glmnet(x = x.train, y = y.train, 
                                type.measure = "mse", family="gaussian",
                                alpha = 0.5, standardize = FALSE)

results <- data.frame(
  y = testing_data$sr,
  ols.pred = as.vector(predict(linear, newdata = testing_data)),
  ridge.pred = as.vector(predict(ridge, newx=x.test, s="lambda.min")),
  lasso.pred = as.vector(predict(lasso, newx=x.test, s="lambda.min")),
  elnet.pred = as.vector(predict(elasticNet, newx=x.test, s="lambda.min"))
)
results$`(y-ols.pred)^2` <- (results$y - results$ols.pred)^2
results$`(y-ridge.pred)^2` <- (results$y - results$ridge.pred)^2
results$`(y-lasso.pred)^2` <- (results$y - results$lasso.pred)^2
results$`(y-elnet.pred)^2` <- (results$y - results$elnet.pred)^2

################################

# Residual Sum of Squares
mapply(results[,6:9],FUN=mean)

################################

coef(linear)

coef(lasso, s="lambda.min")
coef(lasso, s="lambda.1se")

coef(ridge, s="lambda.min")
coef(ridge, s="lambda.1se")

coef(elasticNet, s="lambda.min")
coef(elasticNet, s="lambda.1se")

################################

# Linear R-Squared
sum(results["(y-ols.pred)^2"])/sum((results$y-mean(results$y))^2)

# Ridge R-Squared
sum(results["(y-ridge.pred)^2"])/sum((results$y-mean(results$y))^2)

# Lasso R-Squared
sum(results["(y-lasso.pred)^2"])/sum((results$y-mean(results$y))^2)

# ElasticNet R-Squared
sum(results["(y-elnet.pred)^2"])/sum((results$y-mean(results$y))^2)


