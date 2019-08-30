#!/usr/bin/env python3

import os, sys, bz2, uuid, pickle, json, connexion
import pandas as pd
import numpy as np

from joblib import load

# models folder
UPLOAD_MODELS_FOLDER = "models"

# load default model
DEFAULT_MODEL_FILE = os.path.join(UPLOAD_MODELS_FOLDER, 'model.joblib')

global SERVICE_CONFIG
SERVICE_CONFIG = {
  "model": load(DEFAULT_MODEL_FILE)
}

# predict api
def predict_api(data: str) -> str:
  dataframe_json = data['dataframe_json']
  #print("dataframe_json:\n", dataframe_json)
  dataframe = pd.read_json(dataframe_json, orient='values')
  predictions = predict(dataframe)
  result = np.array2string(predictions)
  return json.dumps(result)

def predict(dataframe):
  global SERVICE_CONFIG
  model = SERVICE_CONFIG['model']
  print("model:\n", model)
  print("dataframe:\n", dataframe)
  predictions = model.predict(dataframe.values)
  return predictions

# deploy api
def deploy_api():
  modelfile = connexion.request.files['modelfile']
  print("modelfile:\n", modelfile)
  model_id = str(uuid.uuid4())
  filename = model_id + ".joblib"
  model_file_path = os.path.join(UPLOAD_MODELS_FOLDER, filename)
  print("model_file_path:\n", model_file_path)
  modelfile.save(model_file_path)
  return "Model deployed"

def load_model(model_file_path):
  from joblib import load
  global SERVICE_CONFIG
  model = load(model_file_path)
  SERVICE_CONFIG['model'] = model
  print("model:\n", model)

def setup():
  print("Initializing with a default model")
  print("DEFAULT_MODEL_FILE:\n", DEFAULT_MODEL_FILE)
  load_model(DEFAULT_MODEL_FILE)

# main
if __name__ == '__main__':  
  setup()
  app = connexion.FlaskApp(__name__, port=9090, specification_dir='swagger/')
  app.add_api('ml_service-api.yaml', arguments={'title': 'Machine Learning Model Service'})
  app.run()
