#!/usr/bin/env python3

import os, sys, bz2, uuid, pickle, json, connexion
import pandas as pd
import numpy as np

from joblib import load
from binascii import hexlify

# api key
API_KEY = hexlify(os.urandom(16)).decode() # str(uuid.uuid4())
#API_KEY = "376d873c859d7f9f268e1b9be883745b"
# tokens
TOKENS = {
  'user'  : API_KEY,
  #'admin' : hexlify(os.urandom(16)).decode()
}

# models folder
MODELS_FOLDER = "models"
# load default model
MODEL_FILE_PATH = os.path.join(MODELS_FOLDER, 'last_model.joblib')

# get token api
def get_token_api() -> str:
  user = connexion.request.form["user"]
  token = TOKENS.get(user)
  if not token:
    return "Invalid user"
  else:
    print('Your token is: {uid}'.format(uid=token))
    return token

#def get_token_api(user) -> str:
#  print('Your token is: {uid}'.format(uid=user))
#  return user

# token info api
#def token_info_api(user) -> dict:
#  #print('user: ', user)
#  token = TOKENS.get(user)
#  #print('token: ', token)
#  if not token:
#    ret = None
#  else:
#    ret = {'uid': token, 'scope': ['uid']}
#  #print('ret: ', ret)
#  return ret

# predict api
def predict_api(data: str) -> str:
  #print("predict_api.data: ", data)
  api_token = data['api_token']
  result = ""
  if api_token == API_KEY:
    dataframe_json = data['dataframe_json']
    #print("dataframe_json:\n", dataframe_json)
    dataframe = pd.read_json(dataframe_json, orient='values')
    predictions = predict(dataframe)
    result = np.array2string(predictions)
  return json.dumps(result)

def predict(dataframe):
  global MODEL_FILE_PATH
  model = load(MODEL_FILE_PATH)
  #print("model:\n", model)
  #print("dataframe:\n", dataframe)
  predictions = model.predict(dataframe.values)
  #print("predictions:\n", predictions)
  return predictions

# deploy api
def deploy_api() -> str:
  api_token = connexion.request.form["api_token"]
  if api_token == API_KEY:
    global MODEL_FILE_PATH
    model_file = connexion.request.files['model_file']
    #print("model_file:\n", model_file)
    #print("model_file_path:\n", MODEL_FILE_PATH)
    model_file.save(MODEL_FILE_PATH)
    return "Model deployed"
  else:
    return "Invalid token"

def setup():
  print("Initializing with a default model")
  print("MODEL_FILE: ", MODEL_FILE_PATH)
  print("API_KEY: ", API_KEY)

# main
if __name__ == '__main__':  
  setup()
  app = connexion.FlaskApp(__name__, port=9090, specification_dir='swagger/')
  app.add_api('ml_service-api.yaml', arguments={'title': 'Machine Learning Model Service'})
  app.run()
