# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Load necessary libraries for data handling, metrics, and visualization.

2. **Load Data**: Read the dataset using `pd.read_csv()` and display basic information.

3. **Initialize Parameters**: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4. **Gradient Descent**: Perform iterations to update `m` and `c` using gradient descent.

5. **Plot Error**: Visualize the error over iterations to monitor convergence of the model.

## Program and Output:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:    Suresh S
RegisterNumber:  212223040215
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def multivariate_linear_regression(X1, Y, learning_rate=0.01, num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-Y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta
dataset = pd.read_csv("/content/50_Startups.csv")
print(dataset.head())
print(dataset.tail())

```

<img width="800" alt="image" src="https://github.com/user-attachments/assets/618a38a2-24fc-4bdc-a075-6e61dc918a3a">


```
dataset.info()
```

<img width="800" alt="image" src="https://github.com/user-attachments/assets/e6c80c73-3326-4ac0-8b7d-315b97ccc6b4">


```
dataset.describe()
```

<img width="800" alt="image" src="https://github.com/user-attachments/assets/723c0d40-2355-44c0-8800-18f8db2079fc">


```
x=dataset.iloc[:,:-2].values
y=(dataset.iloc[:,-1].values).reshape(-1,1)
```



```
print(x)
```

<img width="950" alt="image" src="https://github.com/user-attachments/assets/306b25d6-c44c-4588-9c4b-30f91ed62adc">





```
print(y)
```
<img width="600" alt="image" src="https://github.com/user-attachments/assets/904263a7-226f-4569-add9-f206542b141b">


```
Scaler=StandardScaler()
x1=x.astype(float)
x1_Scaled=Scaler.fit_transform(x1)
y1_Scaled=Scaler.fit_transform(y)
print(x1_Scaled)
print(y1_Scaled)
```
<img width="750" alt="image" src="https://github.com/user-attachments/assets/3e4f179e-2437-467a-a446-a0b8388215a3">

<img width="750" alt="image" src="https://github.com/user-attachments/assets/9f3d88ef-d71e-4cf2-9cc0-ae9caed97c9b">

```
theta=multivariate_linear_regression(x1_Scaled,y1_Scaled)
print(theta)
```
<img width="750" alt="image" src="https://github.com/user-attachments/assets/38dbe638-d0b4-4d07-bfd9-ed74415cc670">


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
