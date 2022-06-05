import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('IrisPredictor.mdl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sepalLength = float(request.form['sepalLength'])
    sepalWidth = float(request.form['sepalWidth'])
    petalLength = float(request.form['petalLength'])
    petalWidth = float(request.form['petalWidth'])
    
    data = {'sepal_length':[sepalLength], 'sepal_width':[sepalWidth], 'petal_length':[petalLength], 'petal_width':[petalWidth]}
    finalFeatures = pd.DataFrame(data) 
    prediction = model.predict(finalFeatures)[0]

    return render_template('index.html', prediction_text='Iris Flower species predicted is     {}'.format(prediction))
   
    

if __name__ == "__main__":
    app.run(debug=True)