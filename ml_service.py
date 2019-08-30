#!/usr/bin/env python3

import sys, bz2, uuid, pickle, json, connexion
import pandas as pd
import numpy as np

from joblib import load

# load default model
model = load('models/model.joblib')


def predict(data: str) -> str:
  dataframe_json = data['dataframe_json']
  #print("dataframe_json:\n", dataframe_json)
  dataframe = pd.read_json(dataframe_json, orient='values')
  print("dataframe:\n", dataframe)
  prediction = np.array2string(model.predict(dataframe.values))
  return json.dumps(prediction)


if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=9090, specification_dir='swagger/')
    app.add_api('ml_service-api.yaml', arguments={'title': 'Machine Learning Model Service'})
    app.run()
