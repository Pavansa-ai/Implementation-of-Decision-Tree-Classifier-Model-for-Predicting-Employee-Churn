# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
   
2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.PAVAN
RegisterNumber: 212225040296

import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
print("data.info():")
data.info()
print("isnull() and sum():")
data.isnull().sum()
print("data value counts():")
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
*/


```

## Output:
<img width="1030" height="192" alt="Screenshot 2026-02-26 111245" src="https://github.com/user-attachments/assets/6315f190-d04c-4a82-a0f3-f0999b14184c" />

<img width="649" height="385" alt="Screenshot 2026-02-26 111252" src="https://github.com/user-attachments/assets/c0ee9d43-a85a-4135-aa54-f514032485f2" />
<img width="548" height="309" alt="Screenshot 2026-02-26 111302" src="https://github.com/user-attachments/assets/a7101397-a3d6-4f8d-a1eb-22bee6d3dbef" />
<img width="1028" height="187" alt="Screenshot 2026-02-26 111309" src="https://github.com/user-attachments/assets/1ff5b6c8-b0f3-4e9a-ae3c-64cc58d501c1" />
<img width="990" height="181" alt="Screenshot 2026-02-26 111314" src="https://github.com/user-attachments/assets/7823c9a4-7da9-4cab-b9cd-cb1aaddff653" />
<img width="541" height="69" alt="Screenshot 2026-02-26 111320" src="https://github.com/user-attachments/assets/a905660d-25cd-4adf-bdde-13bb5292a963" />
<img width="1029" height="109" alt="Screenshot 2026-02-26 111331" src="https://github.com/user-attachments/assets/392980ff-1daa-4c8f-83ef-7dbf57f8c58d" />
<img width="807" height="593" alt="Screenshot 2026-02-26 111338" src="https://github.com/user-attachments/assets/057980e4-e32a-4fbd-b081-2405db81c9e2" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
