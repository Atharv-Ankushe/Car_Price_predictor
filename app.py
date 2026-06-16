import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the Decision Tree Regressor model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "decision_pkl.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Attractive Dashboard Single-Page UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Vehicle Valuation Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
            min-height: 100vh;
        }
        .navbar-brand {
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #38bdf8 !important;
        }
        .main-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 1.25rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.3);
        }
        .form-label {
            font-weight: 600;
            font-size: 0.875rem;
            color: #94a3b8;
        }
        .form-control, .form-select {
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #ffffff;
            border-radius: 0.5rem;
            padding: 0.625rem 1rem;
            transition: all 0.2s ease-in-out;
        }
        .form-control:focus, .form-select:focus {
            background-color: #0f172a;
            border-color: #38bdf8;
            color: #ffffff;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25);
        }
        .btn-predict {
            background: linear-gradient(135deg, #38bdf8 0%, #2563eb 100%);
            border: none;
            color: white;
            font-weight: 600;
            border-radius: 0.5rem;
            padding: 0.75rem;
            transition: transform 0.2s, opacity 0.2s;
        }
        .btn-predict:hover {
            transform: translateY(-1px);
            opacity: 0.95;
        }
        .result-box {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 0.75rem;
        }
    </style>
</head>
<body>

    <nav class="navbar navbar-expand-lg navbar-dark pt-4">
        <div class="container">
            <a class="navbar-brand" href="#"><i class="fa-solid fa-car-rocket me-2"></i>VALUAI.</a>
        </div>
    </nav>

    <div class="container my-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="main-card p-4 p-md-5">
                    <h2 class="text-center fw-bold mb-2">Predict Valuation</h2>
                    <p class="text-center text-muted mb-4">Input details below to dynamically query your decision tree estimator model.</p>
                    
                    {% if prediction_text %}
                    <div class="result-box p-4 text-center mb-4">
                        <span class="d-block text-uppercase small text-success fw-bold tracking-wider mb-1">Estimated Value</span>
                        <h1 class="fw-extrabold text-success m-0">{{ prediction_text }}</h1>
                    </div>
                    {% endif %}

                    <form action="/predict" method="POST">
                        <div class="row g-4">
                            <div class="col-md-6">
                                <label class="form-label"><i class="fa-solid fa-calendar me-1"></i> Production Year</label>
                                <input type="number" class="form-control" name="year" placeholder="e.g. 2022" required min="1900" max="2027" value="{{ inputs.year if inputs else '' }}">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label"><i class="fa-solid fa-gauge me-1"></i> Engine Size (Liters)</label>
                                <input type="number" step="0.1" class="form-control" name="engine_size" placeholder="e.g. 2.0" required value="{{ inputs.engine_size if inputs else '' }}">
                            </div>
                            <div class="col-md-12">
                                <label class="form-label"><i class="fa-solid fa-road me-1"></i> Current Mileage</label>
                                <input type="number" class="form-control" name="mileage" placeholder="e.g. 35000" required min="0" value="{{ inputs.mileage if inputs else '' }}">
                            </div>

                            <div class="col-md-6">
                                <label class="form-label"><i class="fa-solid fa-industry me-1"></i> Manufacturer / Make</label>
                                <select class="form-select" name="make" required>
                                    <option value="0" {% if inputs and inputs.make == '0' %}selected{% endif %}>Toyota</option>
                                    <option value="1" {% if inputs and inputs.make == '1' %}selected{% endif %}>Honda</option>
                                    <option value="2" {% if inputs and inputs.make == '2' %}selected{% endif %}>Ford</option>
                                    <option value="3" {% if inputs and inputs.make == '3' %}selected{% endif %}>BMW</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label"><i class="fa-solid fa-car me-1"></i> Body Model Category</label>
                                <select class="form-select" name="model" required>
                                    <option value="0" {% if inputs and inputs.model == '0' %}selected{% endif %}>Sedan</option>
                                    <option value="1" {% if inputs and inputs.model == '1' %}selected{% endif %}>SUV</option>
                                    <option value="2" {% if inputs and inputs.model == '2' %}selected{% endif %}>Hatchback</option>
                                    <option value="3" {% if inputs and inputs.model == '3' %}selected{% endif %}>Coupe</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label"><i class="fa-solid fa-gas-pump me-1"></i> Fuel Type</label>
                                <select class="form-select" name="fuel_type" required>
                                    <option value="0" {% if inputs and inputs.fuel_type == '0' %}selected{% endif %}>Petrol</option>
                                    <option value="1" {% if inputs and inputs.fuel_type == '1' %}selected{% endif %}>Diesel</option>
                                    <option value="2" {% if inputs and inputs.fuel_type == '2' %}selected{% endif %}>Electric</option>
                                    <option value="3" {% if inputs and inputs.fuel_type == '3' %}selected{% endif %}>Hybrid</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label"><i class="fa-solid fa-gears me-1"></i> Transmission</label>
                                <select class="form-select" name="transmission" required>
                                    <option value="0" {% if inputs and inputs.transmission == '0' %}selected{% endif %}>Manual</option>
                                    <option value="1" {% if inputs and inputs.transmission == '1' %}selected{% endif %}>Automatic</option>
                                    <option value="2" {% if inputs and inputs.transmission == '2' %}selected{% endif %}>Semi-Auto</option>
                                </select>
                            </div>

                            <div class="col-md-12 pt-2">
                                <button type="submit" class="btn btn-predict w-100 text-uppercase tracking-wide">
                                    <i class="fa-solid fa-wand-magic-sparkles me-2"></i>Calculate Value
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form
    
    # Structure inputs sequentially to match your model's expected features
    features = [
        float(form_data['make']),
        float(form_data['model']),
        float(form_data['year']),
        float(form_data['engine_size']),
        float(form_data['mileage']),
        float(form_data['fuel_type']),
        float(form_data['transmission'])
    ]
    
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    prediction_text = f"${output:,}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text, inputs=form_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
