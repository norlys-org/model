from flask import Flask, jsonify
from flask_cors import CORS
from norlys.rendering import create_matrix
import random
import config

app = Flask(__name__)
CORS(app)

@app.route('/map', methods=['GET'])
def get_map():
	scores = {}
	for key in config.STATIONS:
		scores[key] = (random.randint(0, 9), '')
	
	return create_matrix(scores)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)