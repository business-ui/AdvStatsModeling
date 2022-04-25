# Cars dataset with features including make, model, year, engine, and other properties of the car used to predict its price.
library("caret")
library("ggrepel")
library("scales")
library("dplyr")
library("tidyverse")
library("glmnet")
library("fastDummies")
library("lubridate")
library("parallel")
library("tibble")
`%nin%` = Negate(`%in%`)

setwd(choose.dir())

#To load functions and save space
source("project-functions.R")
source("project-preprocessing.R")

input <- read.csv("car_prices.csv")

preprocessed.dfs <- CLEAN(input)

df <- preprocessed.dfs$df
baseline.df <- preprocessed.dfs$baseline.df

rm(list = c(deparse(substitute(input)), 
            deparse(substitute(preprocessed.dfs))))

set.seed(42)
subset_rows <- sample(seq_len(nrow(df)), 20000)

df <- df[subset_rows,]
baseline.df <- baseline.df[subset_rows,]

#### Gaussian Start ####

# get the indices for 85% of the data to train
train_rows <- sample(seq_len(nrow(df)), size = 0.85 * nrow(df))
# # set the train dataset to the 85%
train <- df[train_rows,]
# # set the test dataset to the 15%
test <- df[-train_rows,]

x.train <- train[,names(df) != "sellingprice"]
x.test <- test[,names(df) != "sellingprice"]

# glmnet likes matrices; will scream otherwise
x.train <- as.matrix(x.train)
x.test <- as.matrix(x.test)


y.train <- train[,names(df) == "sellingprice"]
y.test <- test[,names(df) == "sellingprice"]


#### Gaussian OLS ####

lm.model <- lm(sellingprice ~ . , baseline.df[train_rows,])
lm.predicted <- predict.lm(lm.model, baseline.df[-train_rows,], se.fit=TRUE)

lm.predicted$mse <- sum(mean((y.test - lm.predicted$fit)^2))
lm.predicted$mape <- mean(abs((y.test - lm.predicted$fit)/y.test))
lm.predicted$mspe <- mean(((y.test - lm.predicted$fit)/y.test)^2)
lm.predicted$r.squared <- sum((lm.predicted$fit - mean(y.test))^2)/sum((y.test-mean(y.test))^2)
lm.predicted$adj.r.squared <- 1 - ((1 - lm.predicted$r.squared)*(dim(baseline.df)[1]-1))/(dim(baseline.df)[1]-dim(baseline.df)[2] - 1)

# save(lm.model, file="baseline-lm-model.RData")
# save(lm.predicted, file="baseline-lm-results.RData")

lm.results <- select(as.data.frame(lm.predicted), subset=-c("fit","se.fit"))
lm.results <- lm.results[!duplicated(lm.results),]

# write.csv(lm.results, file="linear-gaussian-results.csv")


#### Gaussian ElasticNet ####

cl <- makeCluster(detectCores() - 4)
clusterEvalQ(cl, {library("glmnet")})


# lambda.min minimizes cross-validated error

# lambda.1se favors parsimonious models within 1 standard error of the best model

clusterExport(cl=cl, varlist = c("x.train",
                                 "y.train",
                                 "glmnet_cv_aicc_1se",
                                 "glmnet_cv_aicc_min",
                                 "CROSS_VAL_EXTRAS",
                                 "CROSS_VAL_FUNCTION",
                                 "CROSS_VAL_RESULTS"), envir=environment())

list.of.fits <- parLapply(cl=cl, X=0:10, fun=function(a){return(CROSS_VAL_FUNCTION(a=a,
                                                                                   this.x.train=x.train,
                                                                                   this.y.train=y.train))})
# save(list.of.fits, file="list-of-fits.RData")

clusterExport(cl=cl, varlist = c("list.of.fits",
                                 "x.test",
                                 "y.test"), envir=environment())

results <- parLapply(cl=cl, X=0:10, fun=function(a){return(CROSS_VAL_RESULTS(a2=a,
                                                                             this.x.test=x.test,
                                                                             this.y.test=y.test))})
results <- do.call(rbind,results)
results

# save(results, file="cv-predicted-results.RData")
# write.csv(results, "cv-predicted-results.csv", row.names = FALSE)


stopCluster(cl)

# Mean-Squared Error Plot
plot(list.of.fits[[3]]) # list.of.fits = your cv.glmnet object
# Lambda Plot
plot(list.of.fits[[3]]$glmnet.fit,xvar = "lambda")

# In-Sample RSS
sqrt(sum(lm.model$residuals^2)) # lm.model = your linear model
# In-Sample Regularized RSS
# Using: cv.model$cvsd[cv.model$index[1]] OR
#        cv.model$cvsd[cv.model$index[2]]
results[,c("alpha","SE.min","SE.1se")]

# Remaining Predictors
# Using: coef(cv.model, s="lambda.min") OR coef(cv.model, s="lambda.1se")
results[,c("alpha","betas.min","betas.1se")]

# Out-of-Sample Adj. R-Squared
lm.results$adj.r.squared
# Out-of-Sample Regularized Adj. R-Squared
# Using: cv.model$glmnet.fit$dev.ratio[cv.model$index[1]] OR
#        cv.model$glmnet.fit$dev.ratio[cv.model$index[2]]
results[,c("alpha","adj.r2.min","adj.r2.1se")]



start_time <- Sys.time()
cl <- makeCluster(detectCores() - 4)
clusterEvalQ(cl, {library("glmnet")})


#### Binomial Start ####

# Want to predict if a car is electric/low.emission/hybrid

binomial.baseline.df <- select(baseline.df, -c("hybrid","electric","low.emission"))
binomial.baseline.df$efficient.vehicle <- ifelse(baseline.df$electric == 1 |
                                                   baseline.df$low.emission == 1 |
                                                   baseline.df$hybrid == 1,
                                                 yes=1,
                                                 no=0)
binomial.df <- select(df,-c("hybrid","electric","low.emission"))
binomial.df$efficient.vehicle <- ifelse(df$electric == 1 |
                                          df$low.emission == 1 |
                                          df$hybrid == 1,
                                        yes=1,
                                        no=0)

binomial.baseline.train <- binomial.baseline.df[train_rows,]
train <- binomial.df[train_rows,]
# set the test dataset to the 15%
binomial.baseline.test <- binomial.baseline.df[-train_rows,]
test <- binomial.df[-train_rows,]

x.train <- train[,names(binomial.df) != "efficient.vehicle"]
x.test <- test[,names(binomial.df) != "efficient.vehicle"]

# glmnet likes matrices; will scream otherwise
x.train <- as.matrix(x.train)
x.test <- as.matrix(x.test)

y.train <- train[,names(binomial.df) == "efficient.vehicle"]
y.test <- test[,names(binomial.df) == "efficient.vehicle"]

#### Binomial GLM ####

binomial.baseline.glm <- glm(formula = efficient.vehicle ~ .,
              data=binomial.baseline.train,
              family = "binomial")


binomial.baseline.glm.predicted <- predict.glm(object = binomial.baseline.glm,
                             newdata = select(binomial.baseline.test, -c("efficient.vehicle")),
                             type="response",
                             se.fit=TRUE)
binomial.baseline.glm.predicted$class.prediction[which(binomial.baseline.glm.predicted$fit < 0.5)] <- 0
binomial.baseline.glm.predicted$class.prediction[which(binomial.baseline.glm.predicted$fit >= 0.5)] <- 1

# 
binomial.baseline.glm.predicted$confusionMatrix <- confusionMatrix(
  data = as.factor(
    binomial.baseline.glm.predicted$class.prediction
    ),
  reference = as.factor(
    binomial.baseline.test$efficient.vehicle
    )
  )
save(binomial.baseline.glm.predicted, file="binomial-glm-results.RData")
save(binomial.baseline.glm, file="binomial-glm.RData")
binomial.baseline.glm.results <- confusionMatrix(
  data=as.factor(
    binomial.baseline.glm.predicted$class.prediction
  ),
  reference=as.factor(
    y.test)
)$byClass
binomial.baseline.glm.results <- as.data.frame(list(
  metric=names(binomial.baseline.glm.results),
  score=binomial.baseline.glm.results)
)
rownames(binomial.baseline.glm.results) <- NULL

# write(t(binomial.baseline.glm.results), 
      # file = "linear-binomial-results.csv")


#### Binomial ElasticNet ####

clusterExport(cl=cl, varlist = c("x.train",
                                 "y.train",
                                 "glmnet_cv_aicc_1se",
                                 "glmnet_cv_aicc_min",
                                 "CROSS_VAL_EXTRAS",
                                 "CROSS_VAL_FUNCTION",
                                 "CROSS_VAL_RESULTS",
                                 "confusionMatrix"), envir=environment())
 
binomial.glmnet <- parLapply(cl=cl, X=0:10, fun=function(a){return(CROSS_VAL_FUNCTION(a = a,
                                                                                      glmnet.family="binomial",
                                                                                      this.x.train = x.train,
                                                                                      this.y.train = y.train))})
 

clusterExport(cl=cl, varlist = c("binomial.glmnet", "x.test","y.test"), envir=environment())



binomial.glmnet.results <- parLapply(cl=cl, X=0:10, fun=function(a){return(CROSS_VAL_RESULTS(a2 = a,
                                                                                             this.list.of.fits=binomial.glmnet,
                                                                                             this.x.test = x.test,
                                                                                             this.y.test = y.test))})
binomial.glmnet.results <- do.call(rbind,binomial.glmnet.results)
binomial.glmnet.results <- as.data.frame(binomial.glmnet.results)
binomial.glmnet.results <- lapply(binomial.glmnet.results, as.numeric)
as.data.frame(binomial.glmnet.results)

# save(binomial.glmnet, file="cv-binomial-glmnet.RData")
# save(binomial.glmnet.results, file="cv-binomial-glmnet-results.RData")

# write.csv(binomial.glmnet.results, "cv-predicted-binomial-results.csv", row.names = FALSE)
end_time <- Sys.time()
end_time - start_time

#### Fewer Predictors Tests ####
repeat{
  col.strat.two.thirds <- sample(
    colnames(
      select(df, -c("sellingprice"))
    ),
    ncol(df)/3*2
  )
  col.strat.one.thirds <- sample(col.strat.two.thirds,
    size=length(col.strat.two.thirds)/2
  )
  
  col_matcher <- str_split(col.strat.two.thirds, "_",n = 2,simplify = TRUE)
  if("mmr" %in% col.strat.one.thirds & "age" %in% col.strat.one.thirds & "odometer" %in% col.strat.one.thirds & "condition" %in% col.strat.one.thirds){
    break
  }
}
col_matcher

for (name in names(baseline.df)){
  if (is.factor(baseline.df[,name])){
    curr_levels <- levels(baseline.df[,name])
    for (level in curr_levels){
      if (level %nin% col_matcher[col_matcher[,1]==name,2]){
        baseline.df[,name] <- droplevels(baseline.df[,name], level)
        baseline.df[,name] <- addNA(baseline.df[,name])
        if (nlevels(baseline.df[,name])==1) {
          if (is.na(levels(baseline.df[,name]))){
            baseline.df <- select(baseline.df, -c(name))
          }
        }
      }
    }
      rm(curr_levels)
    
  } else {
    if (length(col_matcher[col_matcher[,1]==name,])==0 & name != "sellingprice"){
      baseline.df <- select(baseline.df, -c(name))
    }
  }
}



df.strat.2.3rds <- select(df, c(all_of(col.strat.two.thirds), "sellingprice"))
df.strat.1.3rds <- select(df, c(all_of(col.strat.one.thirds), "sellingprice"))



#### Gaussian Two-Thirds ####
train <- df.strat.2.3rds[train_rows,]
test <- df.strat.2.3rds[-train_rows,]
x.train <- train[,names(df.strat.2.3rds) != "sellingprice"]
x.train <- as.matrix(x.train)
x.test <- test[,names(df.strat.2.3rds) != "sellingprice"]
x.test <- as.matrix(x.test)
y.train <- train[,names(df.strat.2.3rds) == "sellingprice"]
y.test <- test[,names(df.strat.2.3rds) == "sellingprice"]

#### Gaussian OLS Two-Thirds ####
lm.model <- lm(sellingprice ~ . , baseline.df[train_rows,])
lm.predicted <- predict.lm(lm.model, baseline.df[-train_rows,], se.fit=TRUE)
lm.predicted$mse <- sum(mean((y.test - lm.predicted$fit)^2))
lm.predicted$mape <- mean(abs((y.test - lm.predicted$fit)/y.test))
lm.predicted$mspe <- mean(((y.test - lm.predicted$fit)/y.test)^2)
lm.predicted$r.squared <- sum((lm.predicted$fit - mean(y.test))^2)/sum((y.test-mean(y.test))^2)
lm.predicted$adj.r.squared <- 1 - ((1 - lm.predicted$r.squared)*(dim(baseline.df)[1]-1))/(dim(baseline.df)[1]-dim(baseline.df)[2] - 1)
save(lm.model, file="baseline-lm-model-two-thirds.RData")
save(lm.predicted, file="baseline-lm-results-two-thirds.RData")
lm.results <- select(as.data.frame(lm.predicted), subset=-c("fit","se.fit"))
lm.results <- lm.results[!duplicated(lm.results),]
write.csv(lm.results, file="linear-gaussian-results-two-thirds.csv")


#### Gaussian ElasticNet Two-Thirds ####
cl <- makeCluster(detectCores() - 4)
clusterEvalQ(cl, {library("glmnet")})
clusterExport(cl=cl, varlist = c("x.train",
                                 "y.train",
                                 "glmnet_cv_aicc_1se",
                                 "glmnet_cv_aicc_min",
                                 "CROSS_VAL_EXTRAS",
                                 "CROSS_VAL_FUNCTION",
                                 "CROSS_VAL_RESULTS"), envir=environment())
list.of.fits.two.thirds <- parLapply(cl=cl, X=0:10, fun=function(a){return(CROSS_VAL_FUNCTION(a=a,
                                                                                   this.x.train=x.train,
                                                                                   this.y.train=y.train))})
save(list.of.fits.two.thirds, file="list-of-fits-two-thirds.RData")
clusterExport(cl=cl, varlist = c("list.of.fits.two.thirds",
                                 "x.test",
                                 "y.test"), envir=environment())
results.two.thirds <- parLapply(cl=cl, X=0:10, fun=function(a){return(CROSS_VAL_RESULTS(a2=a,
                                                                             this.list.of.fits = list.of.fits.two.thirds,
                                                                             this.x.test=x.test,
                                                                             this.y.test=y.test))})
results.two.thirds <- do.call(rbind,results.two.thirds)
results.two.thirds
save(results.two.thirds, file="cv-predicted-results-two-thirds.RData")
write.csv(results.two.thirds, "cv-predicted-results-two-thirds.csv", row.names = FALSE)

stopCluster(cl)


col_matcher <- str_split(col.strat.one.thirds, "_",n = 2,simplify = TRUE)


for (name in names(baseline.df)){
  if (is.factor(baseline.df[,name])){
    curr_levels <- levels(baseline.df[,name])
    for (level in curr_levels){
      if (level %nin% col_matcher[col_matcher[,1]==name,2]){
        baseline.df[,name] <- droplevels(baseline.df[,name], level)
        baseline.df[,name] <- addNA(baseline.df[,name])
        if (nlevels(baseline.df[,name])==1) {
          if (is.na(levels(baseline.df[,name]))){
            baseline.df <- select(baseline.df, -c(name))
          }
        }
      }
    }
    rm(curr_levels)
    
  } else {
    if (length(col_matcher[col_matcher[,1]==name,])==0 & name != "sellingprice"){
      baseline.df <- select(baseline.df, -c(name))
    }
  }
}


#### Gaussian One-Thirds ####
train <- df.strat.1.3rds[train_rows,]
test <- df.strat.1.3rds[-train_rows,]
x.train <- train[,names(df.strat.1.3rds) != "sellingprice"]
x.train <- as.matrix(x.train)
x.test <- test[,names(df.strat.1.3rds) != "sellingprice"]
x.test <- as.matrix(x.test)
y.train <- train[,names(df.strat.1.3rds) == "sellingprice"]
y.test <- test[,names(df.strat.1.3rds) == "sellingprice"]

#### Gaussian OLS One-Thirds ####
lm.model <- lm(sellingprice ~ . , baseline.df[train_rows,])
lm.predicted <- predict.lm(lm.model, baseline.df[-train_rows,], se.fit=TRUE)
lm.predicted$mse <- sum(mean((y.test - lm.predicted$fit)^2))
lm.predicted$mape <- mean(abs((y.test - lm.predicted$fit)/y.test))
lm.predicted$mspe <- mean(((y.test - lm.predicted$fit)/y.test)^2)
lm.predicted$r.squared <- sum((lm.predicted$fit - mean(y.test))^2)/sum((y.test-mean(y.test))^2)
lm.predicted$adj.r.squared <- 1 - ((1 - lm.predicted$r.squared)*(dim(baseline.df)[1]-1))/(dim(baseline.df)[1]-dim(baseline.df)[2] - 1)
save(lm.model, file="baseline-lm-model-one-thirds.RData")
save(lm.predicted, file="baseline-lm-results-one-thirds.RData")
lm.results <- select(as.data.frame(lm.predicted), subset=-c("fit","se.fit"))
lm.results <- lm.results[!duplicated(lm.results),]
write.csv(lm.results, file="linear-gaussian-results-one-thirds.csv")


#### Gaussian ElasticNet One-Thirds ####
cl <- makeCluster(detectCores() - 4)
clusterEvalQ(cl, {library("glmnet")})
clusterExport(cl=cl, varlist = c("x.train",
                                 "y.train",
                                 "glmnet_cv_aicc_1se",
                                 "glmnet_cv_aicc_min",
                                 "CROSS_VAL_EXTRAS",
                                 "CROSS_VAL_FUNCTION",
                                 "CROSS_VAL_RESULTS"), envir=environment())
list.of.fits.one.thirds <- parLapply(cl=cl, X=0:10, fun=function(a){return(CROSS_VAL_FUNCTION(a=a,
                                                                                   this.x.train=x.train,
                                                                                   this.y.train=y.train))})
save(list.of.fits.one.thirds, file="list-of-fits-one-thirds.RData")
clusterExport(cl=cl, varlist = c("list.of.fits.one.thirds",
                                 "x.test",
                                 "y.test"), envir=environment())
results.one.thirds <- parLapply(cl=cl, X=0:10, fun=function(a){return(CROSS_VAL_RESULTS(a2=a,
                                                                                        this.list.of.fits = list.of.fits.one.thirds,
                                                                                        this.x.test=x.test,
                                                                                        this.y.test=y.test))})
results.one.thirds <- do.call(rbind,results.one.thirds)
results.one.thirds
save(results.one.thirds, file="cv-predicted-results-one-thirds.RData")
write.csv(results.one.thirds, "cv-predicted-results-one-thirds.csv", row.names = FALSE)



stopCluster(cl)
