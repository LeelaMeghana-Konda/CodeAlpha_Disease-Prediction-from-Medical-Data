import os
import joblib
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

MODEL_DIR = "saved_models"
models = {}
scalers = {}
features = {}

# Load models and scalers
DISEASES = ["heart_disease", "diabetes", "breast_cancer"]
ALGORITHMS = ["svm", "logistic_regression", "random_forest", "xgboost"]

def load_resources():
    for disease in DISEASES:
        models[disease] = {}
        for algo in ALGORITHMS:
            model_path = os.path.join(MODEL_DIR, f"{disease}_{algo}.joblib")
            if os.path.exists(model_path):
                models[disease][algo] = joblib.load(model_path)
                print(f"Loaded {algo} for {disease}")
        
        scaler_path = os.path.join(MODEL_DIR, f"{disease}_scaler.joblib")
        features_path = os.path.join(MODEL_DIR, f"{disease}_features.json")

        if os.path.exists(scaler_path) and os.path.exists(features_path):
            scalers[disease] = joblib.load(scaler_path)
            with open(features_path, 'r') as f:
                features[disease] = json.load(f)
            print(f"Loaded resources for {disease}")
        else:
            print(f"Warning: Base resources for {disease} not found.")

load_resources()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/predict/<disease>', methods=['POST'])
def predict(disease):
    if disease not in models:
        return jsonify({"error": "Disease model not found"}), 404

    try:
        data = request.json
        algo = data.get('algorithm', 'random_forest').lower()
        
        if algo not in models[disease]:
            # Fallback to any available model
            algo = list(models[disease].keys())[0]

        model = models[disease][algo]
        
        # Convert incoming JSON keys to the order expected by the model
        input_data = []
        for feat in features[disease]:
            if feat not in data:
                return jsonify({"error": f"Missing feature: {feat}"}), 400
            input_data.append(float(data[feat]))

        # Transform and predict
        X = np.array([input_data])
        X_scaled = scalers[disease].transform(X)
        
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0].tolist()
        
        # Meta info from data_loader style (optional, but good for UI)
        meta = {
            "heart_disease": {"pos": "Disease", "neg": "No Disease"},
            "diabetes": {"pos": "Diabetic", "neg": "Non-diabetic"},
            "breast_cancer": {"pos": "Malignant", "neg": "Benign"}
        }
        
        result_label = meta[disease]["pos"] if prediction == 1 else meta[disease]["neg"]
        
        return jsonify({
            "disease": disease,
            "algorithm": algo,
            "prediction": prediction,
            "prediction_label": result_label,
            "probability": probabilities[1], # probability of positive class
            "all_probabilities": probabilities
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/features/<disease>', methods=['GET'])
def get_features(disease):
    if disease not in features:
        return jsonify({"error": "Disease features not found"}), 404
    return jsonify({"features": features[disease]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
