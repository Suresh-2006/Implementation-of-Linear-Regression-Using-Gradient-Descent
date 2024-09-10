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
Developed by:    SURESH S
RegisterNumber:  212223040215

*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv("/content/student_scores.csv")
print(dataset.head())
print(dataset.tail())
```

<img width="800" alt="image" src="https://github.com/user-attachments/assets/53cb77bb-109a-4787-b525-3f36a2e37012">

```
dataset.info()
```

<img width="800" alt="image" src="https://github.com/user-attachments/assets/bb6bf5ea-11d2-4d72-8390-3c9e4e8cfbca">

```
dataset.describe()
```

<img width="800" alt="image" src="https://github.com/user-attachments/assets/dabba87b-87bb-4bf9-8d3a-36a5e283b3fc">

```
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
```

<img width="800" alt="image" src="https://github.com/user-attachments/assets/c018ea92-4541-4415-8550-92950559c8e6">

```
x.shape
```

<img width="400" alt="image" src="https://github.com/user-attachments/assets/6ab9d665-2e28-4454-889a-14b12485ae96">

```
y.shape
```

<img width="400" alt="image" src="https://github.com/user-attachments/assets/53f343a0-5102-4dac-9a01-242dafabdd92">


```
m=0
c=0

L=0.001
epochs=5000

n=float(len(x))
error=[]

for i in range(epochs):
  y_pred=m*x+c
  D_m=(-2/n)*sum(x*(y-y_pred))
  D_c=(-2/n)*sum(y-y_pred)
  m=m-L*D_m
  c=c-L*D_c

  error.append(sum(y-y_pred)**2)
print(m,c)
type(error)
print(len(error))
plt.plot(range(0,epochs),error)
```

<img width="800" alt="image" src="https://github.com/user-attachments/assets/b8d8f31d-1e99-4ba3-9a7b-10e0db3a8387">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/3cb9b564-238e-49cf-a3c0-a7d254c24040">
<img width="800" alt="image" src="https://github.com/user-attachments/assets/199a2dad-3535-4db8-bc9b-18463a632db0">


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
