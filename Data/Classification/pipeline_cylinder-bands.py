# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:57:33 2021

@author: RISHBANS
"""
import pandas as pd
dataset = pd.read_csv("cylinder_bands.txt", header=None)

X= dataset.iloc[:, :-1]
y= dataset.iloc[:, -1]

for col in  X.columns[19:]:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X[0] = pd.to_numeric(X[0], errors='coerce')
X[3] = pd.to_numeric(X[15], errors='coerce') 
X[15] = pd.to_numeric(X[15], errors='coerce') 


float_data = X.select_dtypes(include=['float64'])

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#cat_pipeline = make_pipeline(SimpleImputer(),OrdinalEncoder())
#SimpleImpter is for handling missing data in pipeline

obj1_pipeline = make_pipeline(OneHotEncoder(drop='first'))
int_pipeline = make_pipeline(MinMaxScaler(), SelectKBest(k=3,score_func=f_classif))
preprocessor = make_column_transformer(
              (obj1_pipeline, ['salary']),
              (obj2_pipeline, ['dept']),
              (int_pipeline,['number_project', 'average_montly_hours', 'time_spend_company']),
              remainder='passthrough'
)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# building our decision tree classifier and fitting the model
from sklearn.tree import DecisionTreeClassifier
dt_c = DecisionTreeClassifier()
dt_c.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

pred_train = dt_c.predict(X_train)
pred_test = dt_c.predict(X_test)

train_accuracy = accuracy_score(y_train, pred_train)
test_accuracy = accuracy_score(y_test, pred_test)

print(train_accuracy)
print(test_accuracy)


