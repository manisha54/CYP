from flask import Flask, request, render_template
import numpy as np
import joblib

# Flask app setup
app = Flask(__name__)

# Load trained model and scaler
loaded_dtr = joblib.load('dtr.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        # Convert categorical variables into one-hot encoding
        area_features = np.array([1 if Area == area else 0 for area in loaded_scaler.categories_[0]])
        item_features = np.array([1 if Item == item else 0 for item in loaded_scaler.categories_[1]])

        # Create input feature array
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]])
        features = np.hstack((features, area_features.reshape(1, -1), item_features.reshape(1, -1)))

        # Scale input features
        transformed_features = loaded_scaler.transform(features)

        # Predict using the loaded model
        prediction = loaded_dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
