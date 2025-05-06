from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load model (no scaler used)
model = joblib.load('stacked_model.pkl')

# Define the required feature names in order
FEATURES = [
    'age', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    'bmi', 'pulse_pressure', 'map', 'age_bmi_interaction', 'pulse_map_interaction'
]

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        # Validate all required features are present
        missing = [feat for feat in FEATURES if feat not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Prepare input for model
        X = np.array([[data[feat] for feat in FEATURES]])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0, 1]

        return jsonify({'prediction': int(pred), 'probability': round(float(proba), 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return "Cardio Prediction Model API is running!"

if __name__ == '__main__':
    app.run(debug=True)
