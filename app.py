from app.data_utils import write_to_kv
from app.matrix import get_matrix
import requests
import json
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
from datetime import datetime
from config import config
from flask import Flask
from flask_apscheduler import APScheduler
from waitress import serve
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # TODO

app = Flask(__name__)

scheduler = APScheduler()
scheduler.api_enabled = True


@scheduler.task('interval', id='get_matrix', max_instances=1, seconds=60 * 5)
def matrix():
  matrix = get_matrix()
  print(write_to_kv('matrix', json.dumps(matrix)))
  print(write_to_kv('last_updated', datetime.utcnow().isoformat()))

@app.route('/ping')
def ping():
  return 'pong'

if __name__ == "__main__":
  matrix = get_matrix()
  print(write_to_kv('matrix', json.dumps(matrix)))
  print(write_to_kv('last_updated', datetime.utcnow().isoformat()))
  scheduler.init_app(app)
  scheduler.start()
  
  serve(app, host='0.0.0.0', port=8080)