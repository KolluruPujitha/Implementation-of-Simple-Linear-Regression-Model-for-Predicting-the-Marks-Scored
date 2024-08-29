# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1.Start

STEP 2. Import the required libaries and read the dataframe.

STEP 3.Assign hours to X and scores to Y.

STEP 4.Implement training set and test set of the dataframe.

STEP 5.Plot the required graph both for test data and training data.

STEP 6.Find the values of MSE,MAE and RMSE.

STEP 7.End
 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KOLLURU PUJITHA
RegisterNumber: 212223240074
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
![Screenshot 2024-08-28 215449](https://github.com/user-attachments/assets/5e92df36-9908-445b-920a-4b7f1ec80e1e)

![Screenshot 2024-08-28 215549](https://github.com/user-attachments/assets/2b96d9d3-9835-4043-b803-ece43fa6058c)

![Screenshot 2024-08-28 215619](https://github.com/user-attachments/assets/8a72e51b-0daf-4bc1-b97a-cb441c664744)

![Screenshot 2024-08-28 215631](https://github.com/user-attachments/assets/c4384367-17af-44c1-b342-8bb01fe69e87)

![Screenshot 2024-08-28 215653](https://github.com/user-attachments/assets/811cf58f-3165-4c35-a4f2-bc10faa3f4c5)

![Screenshot 2024-08-28 215716](https://github.com/user-attachments/assets/21402d09-0776-4e48-ab32-ff24bfbcab3e)

![Screenshot 2024-08-28 220342](https://github.com/user-attachments/assets/800d6cc0-c178-44c8-8c2d-151e37352a96)

![Screenshot 2024-08-28 220424](https://github.com/user-attachments/assets/fee23a0c-0f8a-4bd8-a8fd-6e4a8f9199ff)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
