from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from capture import start_capture, packet_data  # Assumes you have a capture.py

app = Flask(__name__)

# Load model and encoders
try:
    model = joblib.load('model/rf_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    encoders = joblib.load('model/encoders.pkl')
    selected_features = joblib.load('model/features.pkl')

    PROTOCOL_OPTIONS = encoders['protocol_type'].classes_.tolist()
    SERVICE_OPTIONS = encoders['service'].classes_.tolist()
    FLAG_OPTIONS = encoders['flag'].classes_.tolist()

    print(" Model artifacts loaded successfully.")
except Exception as e:
    print(" Error loading model artifacts:", e)
    raise

# Start packet capture
start_capture()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template(
            'index.html',
            features=selected_features,
            protocol_options=PROTOCOL_OPTIONS,
            service_options=SERVICE_OPTIONS,
            flag_options=FLAG_OPTIONS,
            result=None
        )

    # Handle POST request
    try:
        data = []
        for feature in selected_features:
            value = request.form.get(feature)
            if feature in ['protocol_type', 'service', 'flag']:
                value = encoders[feature].transform([value])[0]
            else:
                value = float(value)
            data.append(value)

        X = np.array([data])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        result = " DoS Attack Detected!" if prediction == 1 else " Normal Traffic"
    except Exception as e:
        result = f" Error: {str(e)}"

    return render_template(
        'index.html',
        features=selected_features,
        protocol_options=PROTOCOL_OPTIONS,
        service_options=SERVICE_OPTIONS,
        flag_options=FLAG_OPTIONS,
        result=result
    )

@app.route('/accuracy')
def accuracy():
    return render_template("result.html")

if __name__ == '__main__':
    app.run(debug=True)
