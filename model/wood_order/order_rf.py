import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.externals import joblib
import joblib

view_dict = {
    "full-shot":0,
    "whole-body":1,
    "above-knee":2,
    "upper-body":3,
    "lower-body":4, 
    "upper-cloth":5,
    "portrait":6,
    "waist":7,
    "detail":8,
    "scene":9
}

direction_dict = {
    "left":0,
    "half-left":1,
    "center":2,
    "half-right":3,
    "right":4,
    "back":5,
    "none":6
}

pose_dict = {
    'stand': 0,
    'sit': 1,
    'walk': 2,
    'spin': 3,
    'none': 4
}

motion_dict = {
    'still': 0,
    'low': 1,
    'high': 2,
    'none': 3
}

feature_dict = {
    'view': 0,
    'direction': 1,
    'pose': 2,
    'motion': 3,
    'none': 4
}

def get_train_data(path):
    csv = pd.read_csv(path)
    size, _ = csv.shape
    x = []
    y = []
    for i in range(size):
        view = view_dict[csv.iloc[i]['view']]
        direction = direction_dict[csv.iloc[i]['direction']]
        pose = pose_dict[csv.iloc[i]['pose']]
        motion = motion_dict[csv.iloc[i]['motion']]
        feature_1 = feature_dict[csv.iloc[i]['feature_1']]
        feature_2 = feature_dict[csv.iloc[i]['feature_2']]
        feature_3 = feature_dict[csv.iloc[i]['feature_3']]
        x.append([view, direction, pose, motion, feature_1])
        y.append(1)
        x.append([view, direction, pose, motion, feature_2])
        y.append(0.5)
        x.append([view, direction, pose, motion, feature_3])
        y.append(0.25)
        have_keys = [csv.iloc[i]['feature_1'], csv.iloc[i]['feature_2'], csv.iloc[i]['feature_3']]
        other_keys = [key for key in feature_dict.keys() if key not in have_keys]
        for key in other_keys:
            x.append([view, direction, pose, motion, feature_dict[key]])
            y.append(0)
    print(len(x), len(y))
    return x, y


import tools.path_tool as pt

def load_model():
    import warnings
    # from sklearn.exceptions import UserWarning
    warnings.filterwarnings(action='ignore', category=UserWarning)
    modelPath = pt.get_path('order_rf.m')
    rfr = joblib.load(modelPath)
    return rfr

if __name__ == "__main__":
    x, y = get_train_data('order.csv')

    # param_test = { 'n_estimators': range(10, 200, 10) }
    # param_test = {'max_depth':range(3,30,2), 'min_samples_leaf':range(2,30,2)}
    # param_test = {'min_samples_split':range(2,45,3), 'min_samples_leaf':range(2,45,3)}
    # gsearch = GridSearchCV(estimator = RandomForestRegressor(n_estimators=110, min_samples_split=17,
    #                         min_samples_leaf=8, max_depth=17, max_features='auto', random_state=10),
    #                         param_grid = param_test, scoring='neg_mean_squared_error', cv=5, n_jobs=4)
    # gsearch.fit(x, y)
    # print(gsearch)
    # summarize the results of the grid search
    # print(gsearch.best_score_)
    # print(gsearch.best_params_)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    rfr = RandomForestRegressor(n_estimators=110, min_samples_split=17,
                            min_samples_leaf=8, max_depth=17, max_features='auto', random_state=10)
    # rfr = RandomForestRegressor(n_estimators=220, min_samples_split=32,
    #                         min_samples_leaf=8, max_depth=13, max_features='auto', random_state=10)
    # rfr = RandomForestRegressor()
    rfr.fit(x, y)
    y_predict = rfr.predict(x_test)
    print(mean_squared_error(y_test, y_predict))
    print(y_test - y_predict)
    y_predict = rfr.predict(x)
    print(mean_squared_error(y, y_predict))
    print(y - y_predict)
    print(y[4])
    print(y_predict[4])
    # joblib.dump(rfr, 'order_rf.m')