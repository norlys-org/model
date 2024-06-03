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

app = Flask(__name__)

scheduler = APScheduler()
scheduler.api_enabled = True

def write_to_kv(key, value):
  url = f"https://api.cloudflare.com/client/v4/accounts/{config['accountID']}/storage/kv/namespaces/{config['namespaceID']}/values/{key}"

  # Create the multipart encoder
  m = MultipartEncoder(
      fields={
          'metadata': json.dumps({
            'date': datetime.utcnow().isoformat()
          }),
          'value': value
      }
  )

  token = os.environ.get('CF_API_TOKEN')
  headers = {
      'Content-Type': m.content_type,
      'Authorization': f'Bearer {token}'
  }

  return requests.put(url, data=m, headers=headers)

@scheduler.task('interval', id='get_matrix', max_instances=1, seconds=60 * 5)
def matrix():
  matrix = get_matrix()
  print(write_to_kv('matrix', json.dumps(matrix)))
  print(write_to_kv('last_updated', datetime.utcnow().isoformat()))

@app.route('/ping')
def ping():
  return 'pong'
  
if __name__ == "__main__":
  scheduler.init_app(app)
  scheduler.start()
  
  serve(app, host='0.0.0.0', port=8080)