# AdvStatsModeling
Work I did for my STA 6257 capstone project regarding Regularized Regression.

### Presentation ![Presentation.pptx](https://github.com/business-ui/AdvStatsModeling/blob/main/Presentation.pptx)
The presentation given to the class and professor explaining the concept of Regularized Regression.

### Simple Example ![simple-example.R](https://github.com/business-ui/AdvStatsModeling/blob/main/simple-example.R)
Summary: I explored the LifeCycleSavings dataset using Gaussian Regression and regularization methods.  
<br />
Models: The models were Ordinary Least Squares (OLS) Regression, Lasso (L_1) Regression, Ridge (L_2) Regression, and Elastic Net Regression.  
<br />
Goal: To demonstrate that despite the increase in variance in-sample with regularization methods, the out-of-sample predictions improve compared to OLS overfitting.  
<br />
Experiments: I compare the total error for each model in-sample, then explore the coefficients of each (Minimum Mean Squared Error and most Parsimonious within 1 Standard Error), and finally the R-squared results for each.  
<br />
Results: Lasso eliminated most or all variables, Ridge brought them asymptotically close to zero, Elastic Net did eliminate a variable, and all three had larger R-squared values than the OLS result for out-of-sample data.  

### Gradients Exploration ![gradients.Rmd](https://github.com/business-ui/AdvStatsModeling/blob/main/gradients.Rmd) & ![gradients.html](https://github.com/business-ui/AdvStatsModeling/blob/main/gradients.html)
Summary: To show how the gradients change for the loss function when using Lasso, Ridge, or Elastic Net.
Methods: 
1. In the Rmd file, I created a Plotly 3D graph in which I plotted the loss gradients for estimating a noisy sine wave.
2. In the Rmd file, I also created the contour plots for the loss functions.
3. In the html file, I used Python to do the same with Matplotlib and was able to show the gradient steps to optimization.

### Project ![project.R](https://github.com/business-ui/AdvStatsModeling/blob/main/project.R)
Summary: I explored the ![Used Cars Dataset](https://www.kaggle.com/tunguz/used-car-auction-prices) which contains over 500,000 rows. I predicted the price of the used car at auction using the same methods as in the simple example above. Given the large dataset, I parallelized the optimization process of the cross-validation. In addition to the price predictions, I explored the explicit dropout of variables and compared the same methods. I also predicted whether cars were electric/low-emission/hybrid vehicles using regularized binomial regression.
Methods: 
- ![project-preprocessing.R](https://github.com/business-ui/AdvStatsModeling/blob/main/project-preprocessing.R) shows the preprocessing methods used. For the sake of a project in academia, I dropped many rows which contained NA or improper values for specific columns.
- ![project.R](https://github.com/business-ui/AdvStatsModeling/blob/main/project.R) I conducted the training of the OLS, Lasso, Ridge, and Elastic Net models using parallel CPU processing. This was done for Gaussian Regression and Binomial Regression. 
- ![project-functions.R](https://github.com/business-ui/AdvStatsModeling/blob/main/project-functions.R) This is the source file for calculating AIC, BIC, # of coefficients, the best R-squared values for desired models, RMSE, MAPE, and MSPE for Gaussian models. It also contains methods for finding the confusion matrix and other binomial regression results.
- ![project-loader.R](https://github.com/business-ui/AdvStatsModeling/blob/main/project-loader.R) Only use this once you've saved your models to .RData files. This will avoid going through all of the model training again.

### Not Explored / Out of Scope
- Outliers and outlier handling
- Heteroskedasticity
- Model Fit with chi-square tests for OLS and Exp(1) for Regularization as per <i>A significance test for the lasso.</i>[[1]](#1)</cite>

<a id="1">[1]</a> Lockhart, R., Taylor, J., Tibshirani, R. J., & Tibshirani, R. (2014). A significance test for the lasso. Annals of statistics, 42(2), 413.
