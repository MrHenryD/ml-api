from flask import Flask, request, abort, jsonify

from config.settings import BOSTON_MODEL_PATH
from models.boston_model import BostonHousingModel

app = Flask(__name__)
model = BostonHousingModel()
model.load(BOSTON_MODEL_PATH)

@app.route('/api/v1/boston', methods = ['GET', 'POST'])
def boston():
	""" Makes a prediction

	POST should include: 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
     'TAX', 'PTRATIO', 'B', 'LSTAT'

	"""
	def validate_post(json_request):
		required = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
		'TAX', 'PTRATIO', 'B', 'LSTAT']

		missing = set(required) - set(json_request.keys())
		if len(missing) > 0:
			return False

		return True

	if request.method == 'GET':
	    abort(404)

	if request.method == 'POST':
		if validate_post(request.json):
			prediction = model.predict_one(request.json)
			return jsonify({'status': 200, 'prediction': float(prediction)})
		else:
			abort(404)

if __name__ == '__main__':
	app.run(debug=True)