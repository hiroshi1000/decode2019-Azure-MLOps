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
    prev_time = time.time()
          
    post = json.loads(raw_data)

    # load and normalize image
    image = np.loadtxt(StringIO(post['image']), delimiter=',') / 255.

    # run model
    with torch.no_grad():
        x = torch.from_numpy(image).float().to(device)
        pred = model(x).detach().numpy()[0]

    # get timing
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)

    payload = {
        'time': inference_time.total_seconds(),
        'prediction': int(np.argmax(pred)),
        'scores': pred.tolist()
    }

    return payload