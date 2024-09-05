from app.data_utils import write_to_d1, write_to_kv
from app.features.quantiles import find_quantile_range
from app.matrix import get_matrix, process_station
import requests
import json
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
from datetime import datetime
from config import config
from flask import Flask
from flask_apscheduler import APScheduler
from waitress import serve
from flask import request
import numpy as np
from pysecs import SECS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # TODO

app = Flask(__name__)

# scheduler = APScheduler()
# scheduler.api_enabled = True


# @scheduler.task('interval', id='get_matrix', max_instances=1, seconds=60 * 5)
# def matrix():
#   matrix, maximum = get_matrix()
#   print(write_to_d1(maximum))
#   print(write_to_kv('matrix', json.dumps(matrix)))
#   print(write_to_kv('last_updated', datetime.utcnow().isoformat()))

@app.route('/predict')
def predict():
  x = request.args.getlist('x')
  y = request.args.getlist('y')
  i = request.args.getlist('i')
  j = request.args.getlist('j')

  R_earth = 6371e3
  # SECS grid setup within the range of input data
  lat, lon, r = np.meshgrid(np.linspace(50, 90),
                            np.linspace(-80, 40),
                            R_earth + 110000, indexing='ij')
  secs_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                              lon.reshape(-1, 1),
                              r.reshape(-1, 1)))

  secs = SECS(sec_df_loc=secs_lat_lon_r)

  # Observation grid matching input data points
  obs_lat_lon_r = np.hstack((y.reshape(-1, 1),
                            x.reshape(-1, 1),
                            np.full((len(x), 1), R_earth)))

  B_obs = np.zeros((1, len(obs_lat_lon_r), 3))
  B_obs[0, :, 0] = i
  B_obs[0, :, 1] = j

  secs.fit(obs_loc=obs_lat_lon_r, obs_B=B_obs, epsilon=0.1)
  # lat_pred, lon_pred, r_pred = np.meshgrid(np.linspace(50, 85, 100),
  #                                         np.linspace(-80, 40, 200),
  #                                         R_earth, indexing='ij')
  lat_pred, lon_pred, r_pred = np.meshgrid(np.linspace(50, 85, 25),
                                          np.linspace(-80, 40, 50),
                                          R_earth, indexing='ij')
  pred_lat_lon_r = np.hstack((lat_pred.reshape(-1, 1),
                              lon_pred.reshape(-1, 1),
                              r_pred.reshape(-1, 1)))
  B_pred = secs.predict(pred_lat_lon_r)

  # Ensure B_pred has the correct shape
  if B_pred.ndim == 2:
      B_pred = B_pred[np.newaxis, ...]

  # Prepare for plotting
  i_pred = B_pred[0, :, 0].reshape(lat_pred.shape)
  j_pred = B_pred[0, :, 1].reshape(lat_pred.shape)

  flat_lon = lon_pred.flatten()
  flat_lat = lat_pred.flatten()
  flat_i = i_pred.flatten()
  flat_j = j_pred.flatten()

  return [{
    'lon': flat_lon[i], 
    'lat': flat_lat[i], 
    'i': flat_i[i], 
    'j': flat_j[i]
    } for i in range(len(i_pred.flatten()))]

if __name__ == "__main__":
  # matrix, maximum = get_matrix()
  # # print(write_to_d1(maximum))
  # # print(write_to_kv('matrix', json.dumps(matrix)))
  # # print(write_to_kv('last_updated', datetime.utcnow().isoformat()))
  # # scheduler.init_app(app)
  # # scheduler.start()
  
  serve(app, host='0.0.0.0', port=8080)