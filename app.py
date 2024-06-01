from app.matrix import get_matrix
import requests
import json
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
from datetime import datetime
from config import config
from flask import Flask

app = Flask(__name__)

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

@app.route('/map', methods=['GET'])
def map():
  matrix = get_matrix()
  write_to_kv('matrix', matrix)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)