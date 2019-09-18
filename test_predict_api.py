#!/usr/bin/env python3

import sys, bz2, uuid, requests, json, pickle
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# defining the api-endpoint  
API_PREDICT_ENDPOINT = 'http://localhost:9090/api/predict'

# your API key here 
API_KEY = "376d873c859d7f9f268e1b9be883745b"

# load the data set
iris = load_iris()
dataframe_load = pd.DataFrame(iris.data)
dataframe_load.columns = iris.feature_names 
data_label = iris.target
dataframe = dataframe_load.assign(LABEL=data_label)

_, dataframe_test = train_test_split(dataframe, test_size=0.1)
dataframe_test  = dataframe_test.reset_index(drop=True)

LABEL_COLUMN = 'LABEL'
columns = [LABEL_COLUMN]

X_test = dataframe_test.drop(columns, axis=1, inplace=False)
y_test = dataframe_test.filter(columns, axis=1)
y_true = y_test.values.ravel()

# convert the dataframe to JSON
dataframe_json = X_test.to_json(orient='values')
print("dataframe_json:\n", dataframe_json)
#print("y_true:\n", y_true)

# send the request and get the results
#headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8', "Authorization": "Bearer user"}
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
data = {
  'dataframe_json': dataframe_json,
  'api_token': API_KEY
}
data_json = json.dumps(data)
req = requests.post(API_PREDICT_ENDPOINT, data=data_json, headers=headers)
#print(req, req.text)

predictions = json.loads(req.text)
#print("predictions:\n", predictions)

y_pred = np.fromstring(predictions[2:-2], dtype=int, sep=' ')
print("y_pred:\n",y_pred)
print("y_true:\n", y_true)

acc = accuracy_score(y_true=y_true, y_pred=y_pred)
print("accuracy_score:", acc)
