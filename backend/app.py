from flask import Flask, send_file, jsonify
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import shap
import json

app = Flask(__name__)
model = keras.models.load_model('model/trained-regressor')
# TODO Allow custom datasets
datapath = 'data/dataset.csv'
y_label_index = 8

@app.route('/')
def hello_world():
    response = jsonify({"lastmessage": "Goodbye",})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/explain')
def explain():
    # Load dataset
    data = pd.read_csv(datapath)   # dataset location
    i = y_label_index              # index of label

    # Select labels
    X = data.drop(data.columns[i],axis=1)
    y = data.iloc[:, 8]
    X_display, y_display = X, y
    explainer = shap.KernelExplainer(f, X.iloc[:50, :])
    shap_values = explainer.shap_values(X.iloc[299, :], nsamples=500)
    shap.force_plot(explainer.expected_value, shap_values, X_display.iloc[299, :], show=False, matplotlib=True)\
        .savefig('scratch.png')
    return send_file('scratch.png', mimetype='image/jpeg')

@app.route('/predict')
def predict():
    # Load dataset
    data = pd.read_csv(datapath)   # dataset location
    i = y_label_index              # index of label

    # Select labels
    X = data.drop(data.columns[i],axis=1)
    y = data.iloc[:, 8]
    X_display, y_display = X, y
    explainer = shap.KernelExplainer(f, X.iloc[:50, :])
    shap_values = explainer.shap_values(X.iloc[299, :], nsamples=500)
    response = jsonify({ # from https://stackoverflow.com/a/44752209
        "expectedvalue": pd.Series(explainer.expected_value).to_json(orient='values'),
        "shapvalues": pd.Series(shap_values).to_json(orient='values'),
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def f(X):
    return model.predict([X[:, i] for i in range(X.shape[1])]).flatten()


if __name__ == '__main__':
    app.run()
