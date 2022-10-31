from flask import Flask, send_file, jsonify
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import shap
import json

app = Flask(__name__)
model = keras.models.load_model('model/trained-regressor')


@app.route('/')
def hello_world():
    response = jsonify({"lastmessage": "Goodbye",})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/explain')
def explain():
    X, y = shap.datasets.adult()
    X_display, y_display = shap.datasets.adult(display=True)
    explainer = shap.KernelExplainer(f, X.iloc[:50, :])
    shap_values = explainer.shap_values(X.iloc[299, :], nsamples=500)
    shap.force_plot(explainer.expected_value, shap_values, X_display.iloc[299, :], show=False, matplotlib=True)\
        .savefig('scratch.png')
    return send_file('scratch.png', mimetype='image/jpeg')

@app.route('/predict')
def predict():
    X, y = shap.datasets.adult()
    X_display, y_display = shap.datasets.adult(display=True)
    explainer = shap.KernelExplainer(f, X.iloc[:50, :])
    shap_values = explainer.shap_values(X.iloc[299, :], nsamples=500)
    return { # from https://stackoverflow.com/a/44752209
        "expectedvalue": pd.Series(explainer.expected_value).to_json(orient='values'),
        "shapvalues": pd.Series(shap_values).to_json(orient='values'),
    }


def f(X):
    return model.predict([X[:, i] for i in range(X.shape[1])]).flatten()


if __name__ == '__main__':
    app.run()
