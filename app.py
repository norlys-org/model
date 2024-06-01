from app.matrix import get_matrix
import requests
import json
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
from datetime import datetime
from config import config

if __name__ == '__main__':
  matrix = get_matrix()

  url = f"https://api.cloudflare.com/client/v4/accounts/{config['accountID']}/storage/kv/namespaces/{config['namespaceID']}/values/{config['matrixKey']}"

  # Get the current date in ISO format
  current_date_iso = datetime.utcnow().isoformat()

  # Define the metadata with the current date
  metadata = {
      'date': current_date_iso
  }

  # Create the multipart encoder
  m = MultipartEncoder(
      fields={
          'metadata': json.dumps(metadata),
          'value': json.dumps(matrix)
      }
  )

  # Retrieve the Cloudflare API token from environment variables
  token = os.environ.get('CF_API_TOKEN')
  headers = {
      'Content-Type': m.content_type,
      'Authorization': f'Bearer {token}'
  }

  # Make the request
  response = requests.put(url, data=m, headers=headers)

  # Print the response
  print(response.text)
