from flask import Flask, request
from waitress import serve
import numpy as np
from pysecs import SECS

app = Flask(__name__)

R_EARTH = 6371e3

def interpolate(x, y, i, j, res_lat=25, res_lon=50):
    # SECS grid setup within the range of input data
    lat = np.linspace(50, 85, res_lat, dtype=np.float32)
    lon = np.linspace(-80, 40, res_lon, dtype=np.float32)
    r = R_EARTH + 110000  # Fixed radius value
    secs_lat_lon_r = np.array(np.meshgrid(lat, lon, [r], indexing='ij')).T.reshape(-1, 3).astype(np.float32)

    # Create SECS object once
    secs = SECS(sec_df_loc=secs_lat_lon_r)

    # Observation grid matching input data points
    obs_lat_lon_r = np.vstack((y, x, np.full(len(x), R_EARTH))).T.astype(np.float32)

    B_obs = np.zeros((1, len(obs_lat_lon_r), 3), dtype=np.float32)
    B_obs[0, :, 0] = i
    B_obs[0, :, 1] = j

    # Fit SECS with observations
    secs.fit(obs_loc=obs_lat_lon_r, obs_B=B_obs, epsilon=0.1)

    # Predict grid
    lat_pred, lon_pred = np.meshgrid(lat, lon, indexing='ij')
    pred_lat_lon_r = np.vstack((lat_pred.ravel(), lon_pred.ravel(), np.full(lat_pred.size, R_EARTH))).T.astype(np.float32)
    B_pred = secs.predict(pred_lat_lon_r)

    # Ensure B_pred has the correct shape
    if B_pred.ndim == 2:
        B_pred = B_pred[np.newaxis, ...]

    # Prepare output
    return {
        'lon': lon_pred.flatten().round(2).tolist(),
        'lat': lat_pred.flatten().round(2).tolist(),
        'i': B_pred[0, :, 0].reshape(lat_pred.shape).flatten().round().tolist(),
        'j': B_pred[0, :, 1].reshape(lat_pred.shape).flatten().round().tolist()
    }

@app.route('/predict', methods=['POST'])
def predict():
    body = request.json
    x = np.array(body['x'], dtype=np.float32)
    y = np.array(body['y'], dtype=np.float32)
    i = np.array(body['i'], dtype=np.float32)
    j = np.array(body['j'], dtype=np.float32)

    result = interpolate(x, y, i, j)

    # Respond with the optimized data
    return [{'lon': result['lon'][k], 'lat': result['lat'][k], 'i': result['i'][k], 'j': result['j'][k]}
            for k in range(len(result['i']))]

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)