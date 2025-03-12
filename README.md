# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## 1.Import the standard Libraries.
## 2.Set variables for assigning dataset values.
## 3.Import linear regression from sklearn.
## 4.Assign the points for representing in the graph.
## 5.Predict the regression for marks by using the representation of the graph.
## 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
Program to implement the simple linear regression model for predicting the marks scored.
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset=pd.read_csv('/student_scores.csv')
print(dataset.head())
print(dataset.tail())
dataset.info()
x=dataset.iloc[:,:-1].values #starts from first untill the last before column
print(x)
y=dataset.iloc[:,1].values #only the last column is extracted
print(y)
print(x.shape)
print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Hours vs Score')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()

plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title('Hours vs Score')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show
```

## Developed by:INFANTINA MARIA L
## RegisterNumber: 212223100013


## Output:
![Screenshot 2025-03-12 103153](https://github.com/user-attachments/assets/0f8018d8-2ae9-4c3b-8d82-34ffe1593251)
![Screenshot 2025-03-12 103233](https://github.com/user-attachments/assets/f5543b34-6076-45b4-9b59-ae6f157adbd1)
![Screenshot 2025-03-12 103249](https://github.com/user-attachments/assets/2592917e-fd1a-498d-b251-48766c94e14a)
![Screenshot 2025-03-12 103301](https://github.com/user-attachments/assets/ef8cfc73-9221-495f-a1ea-a73d54b40c6c)
![Screenshot 2025-03-12 103318](https://github.com/user-attachments/assets/d8718fda-76a3-4d15-bd3c-6c7e71844441)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
