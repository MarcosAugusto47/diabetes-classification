import json
import pandas as pd
from flask import Flask, jsonify, request
from utilities import predict_diabetes

app = Flask(__name__)

covariables = ['Pregnancies',
               'Glucose',
               'BloodPressure',
               'SkinThickness',
               'Insulin',
               'BMI',
               'DiabetesPedigreeFunction',
               'Age']

@app.post('/predict') 
def predict():
    data = request.json
    print(data)

    try:
        data = pd.DataFrame(data)
    except ValueError:
        data = pd.DataFrame([data])

    if list(data.columns) == covariables:
        try:
            sample = data.values
        except KeyError:
            return jsonify({'error':'Invalid input'})
        
        predictions = predict_diabetes(sample)
        
        try:
            result = jsonify(predictions)
        except TypeError as e:
            return jsonify({'error':str(e)})
        
        return result
    
    else:
        return jsonify({'error':'Invalid input'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)