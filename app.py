from app.matrix import get_matrix
import requests
import json
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
from datetime import datetime

if __name__ == '__main__':
  matrix = get_matrix()

  account_id = "027c2b0378c6ce9b76e5b5eab615ba04"
  namespace_id = "ed64384b958c48eeb86b922b3c1aebb0"
  matrix_key = "matrix"
  url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}/values/{matrix_key}"

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
