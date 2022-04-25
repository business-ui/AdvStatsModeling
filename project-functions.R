# set.seed(42)
glmnet_cv_aicc_min <- function(fit, lambda = 'lambda.min'){
  whlm <- which(fit$lambda == fit[[lambda]])
  with(fit$glmnet.fit,
       {
         tLL <- nulldev - nulldev * (1 - dev.ratio)[whlm]
         k <- df[whlm]
         n <- nobs
         return(list('AIC' = - tLL + 2 * k + 2 * k * (k + 1) / (n - k - 1),
                     'BIC' = log(n) * k - tLL))
       })
}
############# lambda.1se favors parsimonious models within 1 standard error of the best model #################
glmnet_cv_aicc_1se <- function(fit, lambda = 'lambda.1se'){
  whlm <- which(fit$lambda == fit[[lambda]])
  with(fit$glmnet.fit,
       {
         tLL <- nulldev - nulldev * (1 - dev.ratio)[whlm]
         k <- df[whlm]
         n <- nobs
         return(list('AIC' = - tLL + 2 * k + 2 * k * (k + 1) / (n - k - 1),
                     'BIC' = log(n) * k - tLL))
       })
}

CROSS_VAL_EXTRAS <- function(cv.model){
  
  cv.model$coefficients.min <- coef(cv.model, s="lambda.min")
  cv.model$coefficients.1se <- coef(cv.model, s="lambda.1se")
  
  
  # r-squared values for two desired models
  cv.model$best.r2.min <- cv.model$glmnet.fit$dev.ratio[cv.model$index[1]]
  cv.model$best.r2.1se <- cv.model$glmnet.fit$dev.ratio[cv.model$index[2]]
  
  # RMSE
  cv.model$rmse.min <- sqrt(cv.model$cvm[cv.model$index[1]])
  cv.model$rmse.1se <- sqrt(cv.model$cvm[cv.model$index[2]])
  
  # Standard Error
  cv.model$SE.min <- cv.model$cvsd[cv.model$index[1]]
  cv.model$SE.1se <- cv.model$cvsd[cv.model$index[2]]
  
  # Akaike & Bayesian Information Criterion
  cv.model$xic.min <- glmnet_cv_aicc_min(cv.model)
  cv.model$xic.1se <- glmnet_cv_aicc_1se(cv.model)
  print("Assigned")
  return(cv.model)
}

CROSS_VAL_FUNCTION <- function(a, glmnet.family="gaussian", this.x.train=x.train, this.y.train=y.train){
  model.alpha <- a / 10
  print(model.alpha)
  cv.model<- cv.glmnet(this.x.train, 
                       this.y.train, 
                       type.measure=ifelse(glmnet.family=="gaussian","mse","deviance"), 
                       alpha=model.alpha, 
                       family=glmnet.family, 
                       parallel=TRUE, 
                       nfolds=5)
  cv.model$alpha = model.alpha
  print("there")
  
  ######### The following variable assignments are in the order: min, 1se. #########
  
  # Assign coefficients of the models
  if(glmnet.family=="gaussian"){
    cv.model <- CROSS_VAL_EXTRAS(cv.model)
  }
  return(cv.model)
}

CROSS_VAL_RESULTS <- function(a2, this.x.test=x.test, this.y.test=y.test, this.list.of.fits=list.of.fits){
  
  
  if(this.list.of.fits[[a2+1]]$name=="Binomial Deviance"){
    
    
    binomial.glmnet.results <- data.frame(
      lambda.min = predict(object = this.list.of.fits[[a2+1]],
                            s="lambda.min",
                            newx = this.x.test,
                            type="response",
                            se.fit=TRUE),
      lambda.1se = predict(object = this.list.of.fits[[a2+1]],
                           s="lambda.1se",
                           newx = this.x.test,
                           type="response",
                           se.fit=TRUE)
    )
    
    binomial.glmnet.results$class.prediction.min <- ifelse(binomial.glmnet.results$lambda.min < 0.5,0,1)
    binomial.glmnet.results$class.prediction.1se <- ifelse(binomial.glmnet.results$lambda.1se < 0.5,0,1)
    
    temp.min <- confusionMatrix(data = as.factor(binomial.glmnet.results$class.prediction.min),
                            reference = as.factor(this.y.test))
    names(temp.min$byClass) <- paste0(names(temp.min$byClass), ".min")
    
    temp.1se <- confusionMatrix(data = as.factor(binomial.glmnet.results$class.prediction.1se),
                                reference = as.factor(this.y.test))
    names(temp.1se$byClass) <- paste0(names(temp.1se$byClass), ".1se")
    
    results <- append(data.frame(), c(temp.min$byClass, temp.1se$byClass))
    results$alpha <- a2/10
    
    return (results)
  }
  
  results <- data.frame()
  
  predicted.min <- predict(this.list.of.fits[[a2+1]], newx=this.x.test, s="lambda.min", type="response")
  predicted.1se <- predict(this.list.of.fits[[a2+1]], newx=this.x.test, s="lambda.1se", type="response")
  
  
  r.squared.min <- sum((predicted.min - mean(this.y.test))^2)/sum((this.y.test-mean(this.y.test))^2)
  r.squared.1se <- sum((predicted.1se - mean(this.y.test))^2)/sum((this.y.test-mean(this.y.test))^2)
  
  adj.r2.min <- 1 - ((1 - r.squared.min) * (nrow(this.x.test)))/(nrow(this.x.test)-this.list.of.fits[[a2+1]]$coefficients.min@p[2])
  adj.r2.1se <- 1 - ((1 - r.squared.1se) * (nrow(this.x.test)))/(nrow(this.x.test)-this.list.of.fits[[a2+1]]$coefficients.1se@p[2])
  
  # MAPE
  mape.min <- mean(abs((this.y.test - predicted.min)/this.y.test))
  mape.1se <- mean(abs((this.y.test - predicted.1se)/this.y.test))
  
  # MSPE
  mspe.min <- mean(((this.y.test - predicted.min)/this.y.test)^2)
  mspe.1se <- mean(((this.y.test - predicted.1se)/this.y.test)^2)
  
  
  temp <- data.frame(alpha=a2/10, 
                     rmse.min=this.list.of.fits[[a2+1]]$rmse.min, 
                     rmse.1se=this.list.of.fits[[a2+1]]$rmse.1se, 
                     r2.min=r.squared.min,
                     r2.1se=r.squared.1se, 
                     SE.min=this.list.of.fits[[a2+1]]$SE.min, 
                     SE.1se=this.list.of.fits[[a2+1]]$SE.1se, 
                     adj.r2.min=adj.r2.min,
                     adj.r2.1se=adj.r2.1se,
                     lambda.min=this.list.of.fits[[a2+1]]$lambda.min,
                     lambda.1se=this.list.of.fits[[a2+1]]$lambda.1se,
                     mape.min = mape.min,
                     mape.1se = mape.1se,
                     mspe.min = mspe.min,
                     mspe.1se = mspe.1se,
                     aic.min=this.list.of.fits[[a2+1]]$xic.min$AIC, 
                     bic.min=this.list.of.fits[[a2+1]]$xic.min$BIC, 
                     aic.1se=this.list.of.fits[[a2+1]]$xic.1se$AIC, 
                     bic.1se=this.list.of.fits[[a2+1]]$xic.1se$BIC, 
                     betas.min=this.list.of.fits[[a2+1]]$coefficients.min@p[2], 
                     betas.1se=this.list.of.fits[[a2+1]]$coefficients.1se@p[2])
  results <- rbind(results, temp)
  
  return(results)
}

