from app.matrix import get_matrix
import requests
import json
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
from datetime import datetime
from config import config
import time

def write_to_kv(key, value):
  url = f"https://api.cloudflare.com/client/v4/accounts/{config['accountID']}/storage/kv/namespaces/{config['namespaceID']}/values/{key}"

  # Create the multipart encoder
  m = MultipartEncoder(
      fields={
          'metadata': json.dumps({
            'date': datetime.utcnow().isoformat()
          }),
          'value': json.dumps(value)
      }
  )

  token = os.environ.get('CF_API_TOKEN')
  headers = {
      'Content-Type': m.content_type,
      'Authorization': f'Bearer {token}'
  }

  response = requests.put(url, data=m, headers=headers)
  return response['success']

def task():
  matrix = get_matrix()
  write_to_kv('matrix', matrix)

def run_periodically(interval):
    while True:
        task()
        time.sleep(interval)

if __name__ == "__main__":
    interval = 5 * 60 # 5 minutes in seconds
    run_periodically(interval)