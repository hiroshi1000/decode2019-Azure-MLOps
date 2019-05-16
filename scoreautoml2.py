import json
import numpy as np

from sklearn.externals import joblib
from azureml.core.model import Model
import azureml.train.automl

def init():
    global model
    model_path = Model.get_model_path('mnist-AutoML')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    y_hat = model.predict(data)
    return y_hat.tolist()