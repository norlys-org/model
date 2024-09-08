from flask import Flask
from flask import request
from waitress import serve
import numpy as np
from pysecs import SECS

app = Flask(__name__)
R_earth = 6371e3

def get_secs():
   # Use linspace to create the latitude and longitude vectors
  lat = np.linspace(50, 85)
  lon = np.linspace(-140, 35)
  r = R_earth + 110000  # Constant value for radius

  # Create a grid of lat, lon, and radius directly in a single array
  lat_lon_r = np.array([[lt, ln, r] for lt in lat for ln in lon])

  # Initialize SECS with the grid data
  return SECS(sec_df_loc=lat_lon_r)

def interpolate(x, y, i, j, res_lat = 25, res_lon = 50):
  secs = get_secs()

  # Observation grid matching input data points
  obs_lat_lon_r = np.column_stack((y, x, np.full_like(x, R_earth)))

  # Initialize observation array directly with the correct shape
  B_obs = np.zeros((1, len(obs_lat_lon_r), 3))
  B_obs[0, :, 0] = i
  B_obs[0, :, 1] = j

  # Fit the SECS model
  secs.fit(obs_loc=obs_lat_lon_r, obs_B=B_obs, epsilon=0.1)

  # Prediction grid setup
  lat_pred = np.linspace(50, 85, res_lat)  # Create latitude array
  lon_pred = np.linspace(-140, 35, res_lon)  # Create longitude array
  pred_lat_lon_r = np.array([[lt, ln, R_earth] for lt in lat_pred for ln in lon_pred])  # Combine lat, lon, and r into one array

  # Predict using the SECS model
  B_pred = secs.predict(pred_lat_lon_r)

  # Ensure B_pred has the correct shape
  if B_pred.ndim == 2:
      B_pred = B_pred[np.newaxis, ...]

  return pred_lat_lon_r[:, 1], pred_lat_lon_r[:, 0], B_pred[0, :, 0], B_pred[0, :, 1]

@app.route('/predict', methods=['POST'])
def predict():
  body = request.json

  # Interpolate the data
  flat_lon, flat_lat, flat_i, flat_j = interpolate(
      np.array(body['x'], dtype=np.float32), 
      np.array(body['y'], dtype=np.float32), 
      np.array(body['i'], dtype=np.float32), 
      np.array(body['j'], dtype=np.float32), 
      25, 
      50
  )

  # Round the output arrays using numpy's vectorized operations
  result = [
      {'lon': round(lon, 2), 'lat': round(lat, 2), 'i': int(round(i_val)), 'j': int(round(j_val))}
      for lon, lat, i_val, j_val in zip(flat_lon, flat_lat, flat_i, flat_j)
  ]

  return result

if __name__ == "__main__":
  serve(app, host='0.0.0.0', port=8080)