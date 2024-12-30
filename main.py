from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd
import xgboost

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
    return str("{:,.0f}".format(np.round(prediction,2)))
if __name__=="__main__":
    app.run(debug=True, port=5000)
