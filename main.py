from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd
import xgboost
import locale

app = Flask(__name__)
data=load('data.joblib')
regressor = load('regressor.joblib')
@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():

    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))
    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location,bhk,bath,sqft]],columns = ['location','bhk','bath','total_sqft'])
    prediction = regressor.predict(input)[0] * 1e5

    # Set locale to Indian numbering system
    locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')  # Ensure your system supports this locale

    formatted_prediction = locale.format_string("%.2f", prediction, grouping=True)
    return str(formatted_prediction)
if __name__=="__main__":
    app.run(debug=True, port=5000)