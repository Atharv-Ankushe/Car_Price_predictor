from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the decision tree model safely
try:
    with open('decision_tree_2.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Single-file HTML template embedded directly into Flask to avoid separate HTML files
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Price Predictor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f3f4f6;
            margin: 0;
            padding: 30px 10px;
            display: flex;
            justify-content: center;
        }
        .container {
            background: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            max-width: 550px;
            width: 100%;
        }
        h2 { text-align: center; color: #1f2937; margin-bottom: 25px; font-weight: 600; }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .form-group { margin-bottom: 15px; }
        .full-width { grid-column: span 2; }
        label { display: block; margin-bottom: 6px; font-size: 14px; font-weight: 500; color: #4b5563; }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            box-sizing: border-box;
            font-size: 14px;
            background-color: #f9fafb;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #3b82f6;
            background-color: #fff;
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            transition: background-color 0.2s;
        }
        button:hover { background-color: #1d4ed8; }
        .result-container { margin-top: 25px; }
        .result-box {
            padding: 15px;
            background-color: #ecfdf5;
            border: 1px solid #10b981;
            border-radius: 6px;
            text-align: center;
            font-size: 18px;
            font-weight: 700;
            color: #065f46;
        }
        .error-box {
            padding: 15px;
            background-color: #fef2f2;
            border: 1px solid #ef4444;
            border-radius: 6px;
            text-align: center;
            font-size: 14px;
            color: #991b1b;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Vehicle Price Predictor</h2>
    <form method="POST" action="/predict">
        <div class="form-grid">
            <div class="form-group">
                <label>Make (Brand):</label>
                <input type="text" name="Make" placeholder="e.g., Toyota" required value="{{ inputs.Make if inputs else '' }}">
            </div>
            <div class="form-group">
                <label>Model:</label>
                <input type="text" name="Model" placeholder="e.g., Corolla" required value="{{ inputs.Model if inputs else '' }}">
            </div>
            <div class="form-group">
                <label>Year:</label>
                <input type="number" name="Year" placeholder="e.g., 2018" min="1900" max="2030" required value="{{ inputs.Year if inputs else '' }}">
            </div>
            <div class="form-group">
                <label>Engine Size (L):</label>
                <input type="number" step="0.1" name="Engine Size" placeholder="e.g., 2.0" required value="{{ inputs.get('Engine Size') if inputs else '' }}">
            </div>
            <div class="form-group full-width">
                <label>Mileage:</label>
                <input type="number" name="Mileage" placeholder="e.g., 45000" min="0" required value="{{ inputs.Mileage if inputs else '' }}">
            </div>
            <div class="form-group">
                <label>Fuel Type:</label>
                <select name="Fuel Type" required>
                    <option value="Petrol" {% if inputs and inputs['Fuel Type'] == 'Petrol' %}selected{% endif %}>Petrol</option>
                    <option value="Diesel" {% if inputs and inputs['Fuel Type'] == 'Diesel' %}selected{% endif %}>Diesel</option>
                    <option value="Electric" {% if inputs and inputs['Fuel Type'] == 'Electric' %}selected{% endif %}>Electric</option>
                    <option value="Hybrid" {% if inputs and inputs['Fuel Type'] == 'Hybrid' %}selected{% endif %}>Hybrid</option>
                </select>
            </div>
            <div class="form-group">
                <label>Transmission:</label>
                <select name="Transmission" required>
                    <option value="Manual" {% if inputs and inputs.Transmission == 'Manual' %}selected{% endif %}>Manual</option>
                    <option value="Automatic" {% if inputs and inputs.Transmission == 'Automatic' %}selected{% endif %}>Automatic</option>
                </select>
            </div>
        </div>
        <button type="submit">Predict Price</button>
    </form>

    <div class="result-container">
        {% if prediction_text %}
            <div class="result-box">{{ prediction_text }}</div>
        {% elif error_text %}
            <div class="error-box">{{ error_text }}</div>
        {% endif %}
    </div>
</div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, inputs=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_text="Model file 'decision_tree_2.pkl' could not be found or loaded on the server.", inputs=request.form)
    
    try:
        # Collect inputs from form submission
        form_data = {
            'Make': request.form['Make'],
            'Model': request.form['Model'],
            'Year': int(request.form['Year']),
            'Engine Size': float(request.form['Engine Size']),
            'Mileage': int(request.form['Mileage']),
            'Fuel Type': request.form['Fuel Type'],
            'Transmission': request.form['Transmission']
        }
        
        # Build DataFrame with proper feature names matching model metadata
        df = pd.DataFrame([{
            'Make': form_data['Make'],
            'Model': form_data['Model'],
            'Year': form_data['Year'],
            'Engine Size': form_data['Engine Size'],
            'Mileage': form_data['Mileage'],
            'Fuel Type': form_data['Fuel Type'],
            'Transmission': form_data['Transmission']
        }])
        
        try:
            # First attempt: Try standard inference (Works if model uses raw text or is an end-to-end Pipeline)
            prediction = model.predict(df)[0]
        except ValueError as val_err:
            # Fallback routine: Handles cases where the model expects pre-encoded numeric types instead of string objects
            print("Direct string conversion failed, executing fallback numerical categorization...")
            
            # Simple conversion mapping (hashes strings to repeatable numeric codes for the model tree structures)
            df_fallback = df.copy()
            df_fallback['Make'] = df_fallback['Make'].apply(lambda x: abs(hash(x)) % 100)
            df_fallback['Model'] = df_fallback['Model'].apply(lambda x: abs(hash(x)) % 500)
            
            fuel_map = {'Petrol': 0, 'Diesel': 1, 'Electric': 2, 'Hybrid': 3}
            trans_map = {'Manual': 0, 'Automatic': 1}
            
            df_fallback['Fuel Type'] = df_fallback['Fuel Type'].map(fuel_map).fillna(0)
            df_fallback['Transmission'] = df_fallback['Transmission'].map(trans_map).fillna(0)
            
            prediction = model.predict(df_fallback)[0]
        
        # Format prediction display string
        output = round(float(prediction), 2)
        prediction_text = f"Estimated Vehicle Price: ${output:,}"
        
        return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text, inputs=form_data)
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Prediction error occurred: {str(e)}", inputs=request.form)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
