#!/usr/bin/env python3

import sys, bz2, uuid, requests, json, pickle
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from joblib import dump

# defining the api-endpoint  
API_DEPLOY_ENDPOINT = 'http://localhost:9090/api/deploy'

# your API key here 
#API_KEY = "XXXXXXXXXXXXXXXXX"

# load the data set
iris = load_iris()
dataframe_load = pd.DataFrame(iris.data)
dataframe_load.columns = iris.feature_names 
data_label = iris.target
dataframe = dataframe_load.assign(LABEL=data_label)

dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.33)
dataframe_train = dataframe_train.reset_index(drop=True)
dataframe_test  = dataframe_test.reset_index(drop=True)

LABEL_COLUMN = 'LABEL'
columns = [LABEL_COLUMN]

X_train = dataframe_train.drop(columns, axis=1, inplace=False)
y_train = dataframe_train.filter(columns, axis=1)

X_test = dataframe_test.drop(columns, axis=1, inplace=False)
y_test = dataframe_test.filter(columns, axis=1)

from sklearn.linear_model import LogisticRegression
input_variables = {'solver': 'lbfgs', 'multi_class': 'auto', 'max_iter': 1000}
clf = LogisticRegression(**input_variables)
print("classifier:\n", clf)

clf.fit(X_train.values, y_train.values.ravel())
print("score:\n", clf.score(X_test.values, y_test.values.ravel()))

dump(clf, 'model.joblib')

fin = open('model.joblib', 'rb')
files = {'modelfile': fin}
try:
  req = requests.post(API_DEPLOY_ENDPOINT, files=files)
  print(req.text)
finally:
  fin.close()

