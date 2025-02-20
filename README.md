# Sydney Airbnb Prices Prediction

## Forcasting Problem: 

To determine the rental price of one specific Airbnb, both the host and the customer have significant and difficult responsibilities. 
- For the host, they want to set a reasonable price without sacrificing the amount of profit they can earn. 
- Customers have the right to understand the significant factors influencing the price, and search for alternative options that desire comparable prices. 

The main objective of this project is to predict Airbnb listing prices in Sydney based on their property characteristics. Specifically, several machine learning models including linear regression, random forest and other models will be adopted with the availability of Scikit-learn module library in Python after cleaning the given 2 datasets.

## Evaluation Criteria

As this forecasting problem directly relates to the supervised regression topic, quality indicators such as Mean Squared Error (MSE), R-squared (R^2) and Cross-Validation Score are applied to compare the performance of all possible models. Specifically, the model with the smallest mean squared error will be performed on the test dataset to predict the corresponding prices.