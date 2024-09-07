from flask import Flask
from flask import request
from waitress import serve
import numpy as np
from pysecs import SECS

app = Flask(__name__)

def interpolate(x, y, i, j, res_lat = 25, res_lon = 50):
  R_earth = 6371e3
  # SECS grid setup within the range of input data
  lat, lon, r = np.meshgrid(np.linspace(50, 85),
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
  lat_pred, lon_pred, r_pred = np.meshgrid(np.linspace(50, 85, res_lat),
                                          np.linspace(-80, 40, res_lon),
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
  
  return (flat_lon, flat_lat, flat_i, flat_j)

@app.route('/predict', methods=['POST'])
def predict():
  body = request.json
  x = np.array(body['x'])
  y = np.array(body['y'])
  i = np.array(body['i'])
  j = np.array(body['j'])

  (flat_lon, flat_lat, flat_i, flat_j) = interpolate(x, y, i, j, 37, 75)

  return [{
    'lon': round(flat_lon[i], 2), 
    'lat': round(flat_lat[i], 2), 
    'i': round(flat_i[i]), 
    'j': round(flat_j[i])
  } for i in range(len(flat_i))]

if __name__ == "__main__":
  serve(app, host='0.0.0.0', port=8080)