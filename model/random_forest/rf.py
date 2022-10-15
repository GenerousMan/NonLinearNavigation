import joblib
import tools.path_tool as pt

def load_model():
    modelPath = pt.get_path("knn_trainmodel.m")
    clf = joblib.load(modelPath)
    return clf