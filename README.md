# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Preprocessing & Vectorization: Map labels to binary values (0 and 1) and add a "bias" column of ones to the feature matrix X to allow for an intercept.

2.Probability Transformation: Pass the linear combination of inputs and weights through the Sigmoid function to generate probabilities between 0 and 1.

3.Cost Optimization: Use the Binary Cross-Entropy (Log Loss) function to measure the error and compute gradients to minimize this "cost."

4.Parameter Update: Iteratively refine the weights (theta) by subtracting the gradient scaled by the learning rate (alpha) until the cost stabilizes. 


## Program:

Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: THANZIL HUSSAIN A

RegisterNumber: 212225240169

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Placement_Data (1).csv")

df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})

X = df[['ssc_p', 'mba_p']].values
y = df['status'].values


scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")


plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
```

## Output:

<img width="1022" height="590" alt="image" src="https://github.com/user-attachments/assets/0ee62c84-d969-4684-b874-ccf3834db01d" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

