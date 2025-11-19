############################################################
# 1. Import Libraries
############################################################

import pickle
from flask import Flask
from flask import request
from flask import jsonify


############################################################
# 2. Load Trained Model & DictVectorizer
############################################################

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


############################################################
# 3. Initialize Flask Application
############################################################

app = Flask('product_subscription')


############################################################
# 4. Prediction Endpoint
############################################################

@app.route('/predict', methods=['POST'])
def predict():
    # Read JSON request body
    customer = request.get_json()

    # Transform input using DictVectorizer
    X = dv.transform([customer])

    # Predict probability of default
    y_pred = model.predict_proba(X)[0, 1]
    product_subscription = y_pred >= 0.5

    # Prepare response
    result = {
        'actual_probability': float(y_pred),
        'loan_default_probability': float(product_subscription)
    }

    return jsonify(result)


############################################################
# 5. Main Application Runner
############################################################

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
