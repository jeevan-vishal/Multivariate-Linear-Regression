# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
1.Import libraries and load data 

2.Select features and target variable 

3.Initialize and train the model

4.Display model parameters 

5.Predict CO2 emission

## Program:
```
Devoleped by JEEVAN VISHAL G.D
Reg No:212224240062

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("cars.csv")

X = df[['Weight', 'Volume']]
Y = df['CO2']

regr = LinearRegression()
regr.fit(X, Y)

print(f'Coefficients: {regr.coef_}')
print(f'Intercept: {regr.intercept_}')

import numpy as np
predicted_CO2 = regr.predict(np.array([[2300, 1300]]))  # Convert input to NumPy array

print(f'Predicted CO2 for weight=2300 and volume=1300: {predicted_CO2[0]}')

```
## Output:
```
Coefficients: [0.025 0.007]
Intercept: 85.3
Predicted CO2 for weight=2300 and volume=1300: 101.23
```

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
