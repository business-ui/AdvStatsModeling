# AdvStatsModeling
Work I did for my STA 6257 capstone project regarding Regularized Regression.

### Simple Example ![simple-example.R](https://github.com/business-ui/AdvStatsModeling/blob/main/simple-example.R)
Summary: I explored the LifeCycleSavings dataset using Gaussian Regression and regularization methods.
\newline
Models: The models were Ordinary Least Squares (OLS) Regression, Lasso ($L_1$) Regression, Ridge ($L_2$) Regression, and Elastic Net Regression.
\newline
Goal: To demonstrate that despite the increase in variance in-sample with regularization methods, the out-of-sample predictions improve compared to OLS overfitting.
\newline
Experiments: I compare the total error for each model in-sample, then explore the coefficients of each (Minimum Mean Squared Error and most Parsimonious within 1 Standard Error), and finally the R-squared results for each.
\newline
Results: Lasso eliminated most or all variables, Ridge brought them asymptotically close to zero, Elastic Net did eliminate a variable, and all three had larger R-squared values than the OLS result for out-of-sample data.

### Gradients Exploration ![gradients.Rmd](https://github.com/business-ui/AdvStatsModeling/blob/main/gradients.Rmd) & ![gradients.html](https://github.com/business-ui/AdvStatsModeling/blob/main/gradients.html)
Summary: To show how the gradients change for the loss function when using Lasso, Ridge, or Elastic Net.
Methods: 
1. In the Rmd file, I created a Plotly 3D graph in which I plotted the loss gradients for estimating a noisy sine wave.
2. In the Rmd file, I also created the contour plots for the loss functions.
3. In the html file, I used Python to do the same with Matplotlib and was able to show the gradient steps to optimization.
