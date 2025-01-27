from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("iris_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)[0]
        species = ['Setosa', 'Versicolor', 'Virginica'][prediction]

        return render_template('index.html', prediction_text=f'Iris Species: {species}')
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
