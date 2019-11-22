import pickle
import os

MODEL_FOLDER = os.environ.get('MODEL_FOLDER', 'trained_gan')
DATA_FOLDER = os.environ.get('DATA_FOLDER')
PLOTS_FOLDER = os.environ.get('PLOTS_FOLDER')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def store_artifact(obj, model_name, obj_name, group_name='data'):
    model_artifact_path = DATA_FOLDER + '/' + model_name + '/' + group_name
    ensure_dir(model_artifact_path)
    with open(model_artifact_path + '/' + obj_name + '.pkl', "wb") as f:
        pickle.dump(obj, f)


def load_artifact(model_name, obj_name, group_name='data'):
    model_artifact_path = DATA_FOLDER + '/' + model_name + '/' + group_name
    with open(model_artifact_path + '/' + obj_name + '.pkl', "rb") as f:
        return pickle.load(f)


def get_artifact_path(model_name, obj_name, group_name='data'):
    model_artifact_path = DATA_FOLDER + '/' + model_name + '/' + group_name
    ensure_dir(model_artifact_path)
    return model_artifact_path + '/' + obj_name


def get_model_path(name):
    model_path = MODEL_FOLDER + '/' + name
    ensure_dir(model_path)
    return model_path