import os
import pandas as pd
import joblib
from prediction_model.config import config

def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(file_path)
    return _data

def save_pipeline(pipeline_to_save, file_name):
    save_path = os.path.join(config.SAVE_MODEL_PATH, file_name)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {file_name}")

def load_pipeline(file_name):
    load_path = os.path.join(config.SAVE_MODEL_PATH, file_name)
    model_loaded = joblib.load(load_path)
    print(f"Model has been loaded")
    return model_loaded
