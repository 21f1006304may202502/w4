import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
#from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

data = pd.read_csv("iris.csv")
data.head(5)

train,test = train_test_split(data,test_size=0.4,stratify=data['species'],random_state=42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

mod_dt = DecisionTreeClassifier(max_depth=3,random_state=1)
mod_dt.fit(X_train,y_train)
y_pred = mod_dt.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred,average='macro')
recall = recall_score(y_test,y_pred,average='macro')
f1 = f1_score(y_test,y_pred,average='macro')
print("accuracy:",accuracy)
print("precision:",precision)
print("recall:",recall)
print("f1:",f1)

joblib.dump(mod_dt, "model.joblib")
