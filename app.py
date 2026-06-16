from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the decision tree model
try:
    with open('decision_tree_2.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Single-file HTML template
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
            background-color: #f4f7f6;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            max-width: 500px;
            width: 100%;
        }
        h2 { text-align: center; color: #333; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #666; }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover { background-color: #218838; }
        .result {
            margin-top: 25px;
            padding: 15px;
            background-color: #e2f0d9;
            border-left: 5px solid #28a745;
            border-radius: 4px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #1e5622;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Vehicle Price Predictor</h2>
    <form method="POST" action="/predict">
        <div class="form-group">
            <label>Make (Brand):</label>
            <input type="text" name="Make" placeholder="e.g., Toyota" required>
        </div>
        <div class="form-group">
            <label>Model:</label>
            <input type="text" name="Model" placeholder="e.g., Corolla" required>
        </div>
        <div class="form-group">
            <label>Year:</label>
            <input type="number" name="Year" placeholder="e.g., 2018" required>
        </div>
        <div class="form-group">
            <label>Engine Size (L):</label>
            <input type="number" step="0.1" name="Engine Size" placeholder="e.g., 2.0" required>
        </div>
        <div class="form-group">
            <label>Mileage:</label>
            <input type="number" name="Mileage" placeholder="e.g., 45000" required>
        </div>
        <div class="form-group">
            <label>Fuel Type:</label>
            <select name="Fuel Type" required>
                <option value="Petrol">Petrol</option>
                <option value="Diesel">Diesel</option>
                <option value="Electric">Electric</option>
                <option value="Hybrid">Hybrid</option>
            </select>
        </div>
        <div class="form-group">
            <label>Transmission:</label>
            <select name="Transmission" required>
                <option value="Manual">Manual</option>
                <option value="Automatic">Automatic</option>
            </select>
        </div>
        <button type="submit">Predict Price</button>
    </form>

    {% if prediction_text %}
        <div class="result">
            {{ prediction_text }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="Model file error. Check server logs.")
    
    try:
        # Extract data from form and build DataFrame to keep feature names intact
        input_data = {
            'Make': [request.form['Make']],
            'Model': [request.form['Model']],
            'Year': [int(request.form['Year'])],
            'Engine Size': [float(request.form['Engine Size'])],
            'Mileage': [int(request.form['Mileage'])],
            'Fuel Type': [request.form['Fuel Type']],
            'Transmission': [request.form['Transmission']]
        }
        
        df = pd.DataFrame(input_data)
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Format the output output
        output = round(prediction, 2)
        
        return render_template_string(
            HTML_TEMPLATE, 
            prediction_text=f'Estimated Vehicle Price: ${output:,}'
        )
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
