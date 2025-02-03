# -*- coding: utf-8 -*-
"""
This Python script shows a high-level example of linear regression analysis. 
It generates random independent (X) and dependent (Y) data following a linear relationship with added noise.

The script calculates and prints the correlation coefficient to assess the strength of the linear association. 
It implements a simple linear regression model to predict a specified random value and calculates the prediction interval 
for forecast values at a 95% confidence level.

Finally, the script visualizes the data, regression line, predicted value, and prediction interval on a plot.

This script does not provide a full analysis of all assumptions and diagnostics related to Simple Linear Regression.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t, pearsonr

# Step 1: Generate random X and Y data value
#np.random.seed(0)  # Uncomment for reproducibility
X = np.random.rand(5000, 1) * 10  # Generates 5000 random values for independent variable X, scaled between 0 and 10
noise = np.random.randn(5000, 1)  # Generates random noise to introduce variability
Y = 2.5 * X + 5 + noise  # dependent variable Y. Linear relationship with noise 

# Step 2: Calculate and print the Correlation Coefficient to evaluate the strength of the linear association between X and Y
correlation_coefficient, p_value = pearsonr(X.flatten(), Y.flatten())
print(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}")

# Step 3: Apply Simple Linear Regression
model = LinearRegression() #Initializes a Linear Regression model.
model.fit(X, Y) #Fits the model 
slope = model.coef_[0] #the slope of the regression line
intercept = model.intercept_ #the intercept of the regression line

# Predict the outcome for some random value
random_value = np.array([[5]]) # you can put your number from the range 0 - 10
predicted_value = model.predict(random_value) #Predicts Y for the given random X value

# Step 4: Calculate the prediction interval for the forecast value
confidence_level = 0.95 #Sets the confidence level 95% for the prediction interval.
degrees_of_freedom = len(X) - 2 #number of observations minus the number of parameters estimated (slope and intercept)
t_value = t.ppf((1 + confidence_level) / 2., degrees_of_freedom) #calculating the prediction interval 

# Standard error of the estimate and Prediction interval
se = np.sqrt(np.sum((Y - model.predict(X))**2) / degrees_of_freedom) #the average distance that the observed values fall from the regression line
mean_x = np.mean(X)
n = len(X)
se_pred = se * np.sqrt(1 + 1/n + (random_value - mean_x)**2 / np.sum((X - mean_x)**2)) #the standard error of the prediction
interval = t_value * se_pred #the prediction interval
lower_bound = predicted_value - interval #the actual value is expected to fall with 95% confidence
upper_bound = predicted_value + interval #the actual value is expected to fall with 95% confidence

print(f"Predicted value for random X = {random_value.flatten()[0]}: {predicted_value.flatten()[0]:.2f}")
print(f"95% Prediction Interval: ({lower_bound.flatten()[0]:.2f}, {upper_bound.flatten()[0]:.2f})")

# Step 5: Plot the data
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.scatter(random_value, predicted_value, color='green', label='Predicted value')
plt.errorbar(random_value, predicted_value, yerr=interval, color='green', fmt='o', label='Prediction interval')

plt.xlabel('Independent X')
plt.ylabel('Dependent Y')
plt.title('Linear Regression with Prediction Interval')
plt.legend()
plt.show()
