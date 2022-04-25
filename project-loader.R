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

#### Gaussian Start ####

# get the indices for 85% of the data to train
train_rows <- sample(seq_len(nrow(df)), size = 0.85 * nrow(df))
# set the train dataset to the 85%
train <- df[train_rows,]
# set the test dataset to the 15%
test <- df[-train_rows,]

x.train <- train[,names(df) != "sellingprice"]
x.test <- test[,names(df) != "sellingprice"]

# glmnet likes matrices; will scream otherwise
x.train <- as.matrix(x.train)
x.test <- as.matrix(x.test)


y.train <- train[,names(df) == "sellingprice"]
y.test <- test[,names(df) == "sellingprice"]

#### Gaussian OLS ####

load("baseline-lm-model.RData")
load("baseline-lm-results.RData")

#### Gaussian ElasticNet ####

# lambda.min minimizes cross-validated error

# lambda.1se favors parsimonious models within 1 standard error of the best model
load("list-of-fits.RData")
load("cv-predicted-results.RData")

write.csv(results, "cv-predicted-results.csv", row.names = FALSE)
write.csv(select(as.data.frame(lm.predicted), 
                 subset=-c("fit","se.fit")), 
          file="linear-gaussian-results.csv")


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
# # set the test dataset to the 15%
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

load("cv-binomial-glm.RData")
load("cv-binomial-glm-results.RData")

#### Binomial ElasticNet ####

load("cv-binomial-glmnet.RData")
load("cv-binomial-glmnet-results.RData")

write.csv(binomial.glmnet.results, "cv-predicted-binomial-results.csv", row.names = FALSE)

#### Fewer Predictors Tests ####
col.strat.two.thirds <- sample(
  colnames(
    select(baseline.df, -c("sellingprice"))
  ),
  ncol(baseline.df)/3*2
)
col.strat.one.thirds <- sample(
  colnames(
    select(baseline.df, -c("sellingprice"))
  ),
  ncol(baseline.df)/3
)

baseline.df.strat.2.3rds <- select(baseline.df, c(all_of(col.strat.two.thirds), "sellingprice"))
baseline.df.strat.1.3rds <- select(baseline.df, c(all_of(col.strat.one.thirds), "sellingprice"))

df.strat.2.3rds <- select(baseline.df, c(all_of(col.strat.two.thirds), "sellingprice"))
df.strat.1.3rds <- select(baseline.df, c(all_of(col.strat.one.thirds), "sellingprice"))


df.strat.2.3rds <- dummy_cols(df.strat.2.3rds,
                              select_columns = colnames(
                                df.strat.2.3rds %>% select_if(is.factor)),
                              remove_selected_columns = TRUE)
df.strat.1.3rds <- dummy_cols(df.strat.1.3rds,
                              select_columns = colnames(
                                df.strat.1.3rds %>% select_if(is.factor)),
                              remove_selected_columns = TRUE)

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

load("baseline-lm-model-two-thirds.RData")
load("baseline-lm-results-two-thirds.RData")

#### Gaussian ElasticNet Two-Thirds ####

load("list-of-fits-two-thirds.RData")
load("cv-predicted-results-two-thirds.RData")


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

load("baseline-lm-model-two-thirds.RData")
load("baseline-lm-results-two-thirds.RData")

#### Gaussian ElasticNet One-Thirds ####

load("list-of-fits-one-thirds.RData")
load("cv-predicted-results-one-thirds.RData")




# stopCluster(cl)
