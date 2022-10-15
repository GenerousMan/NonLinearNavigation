import math
import matplotlib as mpl
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from sklearn.externals import joblib #jbolib模块
import joblib

def get_mirro_data(data):
    # width,height
    # pose_1,pose_2,pose_3,pose_4,pose_5,pose_6,pose_7,pose_8,pose_9,pose_10,pose_11,pose_12,pose_13,pose_14,pose_15,pose_16,pose_17,pose_18,pose_19,pose_20,pose_21,pose_22,pose_23,pose_24,pose_25,pose_26,pose_27,pose_28,pose_29,pose_30,pose_31,pose_32,pose_33,pose_34,pose_35,pose_36,pose_37,pose_38,pose_39,pose_40,pose_41,pose_42,pose_43,pose_44,pose_45,pose_46,pose_47,pose_48,pose_49,pose_50,pose_51
    # roi_1,roi_2,roi_3,roi_4
    width, height = data[:2]
    frame_x_center = int(width) / 2
    y1, x1, y2, x2 = data[53:]
    y1 = float(y1)
    x1 = float(x1)
    y2 = float(y2)
    x2 = float(x2)
    x_center = (x2 - x1) / 2
    pose = []
    for i in range(17):
        x, y, prob = data[2+i*3:5+i*3]
        x = float(x)
        y = float(y)
        prob = float(prob)
        x = 2 * x_center - x
        pose += [x, y, prob]
    x1 = int(2 * frame_x_center - x1)
    x2 = int(2 * frame_x_center - x2)
    return [width, height] + pose + [y1, x1, y2, x2]

def preprocess_xy(features, labels):
    x_view = []
    x_direction = []
    y_view = []
    y_direction = []
    for feature in features:
        if (feature[0]=="file_name"):
            continue
        try:
            #print(feature[0])
            feature_video = int(feature[0].split(".")[0])

            x_view.append(feature[2:])
            x_view.append(get_mirro_data(feature[2:]))
            y_view.append(view_label_index[labels[feature_video+1][1]])
            y_view.append(view_label_index[labels[feature_video+1][1]])
            pose = labels[feature_video+1][3]
            direction = labels[feature_video+1][2]
            if pose == 'spin':
            # if pose == 'spin' or pose == 'none' or direction == 'none':
                continue
            x_direction.append(feature[2:])
            x_direction.append(get_mirro_data(feature[2:]))
            y_direction.append(direction_label_index[labels[feature_video+1][2]])
            y_direction.append(opposite_direction_label_index[labels[feature_video+1][2]])
        except ValueError:
            continue

    return np.array(x_view), np.array(x_direction), np.array(y_view), np.array(y_direction)

def read_csv(file_name):
    f = open(file_name, 'r')
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    for row in rows:
        final_list.append(row.split(','))
    return final_list

def train(x, y):
    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)

    # clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)

    # clf2 = RandomForestClassifier(n_estimators=10,max_features=10, max_depth=None,min_samples_split=2, bootstrap=True)
    # clf3 = ExtraTreesClassifier(n_estimators=10, max_features=10, max_depth=None, min_samples_split=2, bootstrap=False)
    clf = ExtraTreesClassifier(oob_score=True, bootstrap=True, n_jobs=4)
    
    '''
    #交叉验证
    scores1 = cross_val_score(clf1, x_train, y_train)
    scores2 = cross_val_score(clf2, x_train, y_train)
    scores3 = cross_val_score(clf3, x_train, y_train)
    print('DecisionTreeClassifier交叉验证准确率为:'+str(scores1.mean()))    
    print('RandomForestClassifier交叉验证准确率为:'+str(scores2.mean()))    
    print('ExtraTreesClassifier交叉验证准确率为:'+str(scores3.mean()))
    '''
    
    # clf1.fit(x_train, y_train)
    # clf2.fit(x_train, y_train)
    # clf3.fit(x_train, y_train)
    # clf.fit(x_train, y_train)
    clf.fit(x, y)
    

    # preds_1 = clf1.predict(x_test)
    # preds_2 = clf2.predict(x_test)
    # preds_3 = clf3.predict(x_test)
    # preds = clf.predict(x_test)

    # print("DecisionTreeClassifier:\n", classification_report(y_test, preds_1))
    # print("RandomForestClassifier\n",classification_report(y_test, preds_2))
    # print("ExtraTreesClassifier\n",classification_report(y_test, preds_3))
    # print("Filnal\n", classification_report(y_test, preds))

    # return clf1, clf2, clf3
    return clf

def get_test_data(test_data, mark_data):
    mark_data = pd.read_csv(mark_data)
    test_data = pd.read_csv(test_data)
    size, _ = mark_data.shape
    test_group = []
    for i in range(size):
        file_name = mark_data.iloc[i]['file_name']
        video_frames = test_data[test_data['file_name']==file_name]
        length, _ = video_frames.shape
        test_unit = []
        for j in range(length):
            frame = video_frames[video_frames['frame_number']==j]
            x_data = np.array(frame)[0][2:].tolist()
            test_unit.append(x_data)
        test_group.append({
            'file_name': file_name,
            'test_unit': test_unit
        })
    return test_group

def view_predict(test_data, test_meta):
    target_names = ["full-shot", "whole-body", "above-knee", "upper-body", "lower-body", "upper-cloth", "portrait", "waist", "detail", "scene"]
    clf = joblib.load('ml_models/view_clf.m')
    test_group = get_test_data(test_data, test_meta)
    for unit in test_group:
        file_name = unit['file_name']
        test_unit = unit['test_unit']
        test_result = pd.Series(clf.predict(test_unit))
        print(file_name, target_names[test_result.mode()[0]])
        print(pd.value_counts(test_result))
        print(np.array(test_result).tolist())

def direction_predict(test_data, test_meta):
    target_names = ["left", "half-left", "center", "half-right", "right", "back", "none"]
    clf = joblib.load('ml_models/direction_clf.m')
    test_group = get_test_data(test_data, test_meta)
    for unit in test_group:
        file_name = unit['file_name']
        test_unit = unit['test_unit']
        test_result = pd.Series(clf.predict(test_unit))
        print(file_name, target_names[test_result.mode()[0]])
        print(pd.value_counts(test_result))
        print(np.array(test_result).tolist())

import tools.path_tool as pt

def load_model():
    import warnings
    # from sklearn.exceptions import UserWarning
    warnings.filterwarnings(action='ignore', category=UserWarning)
    modelPath = pt.get_path('direction_clf.m')
    clf = joblib.load(modelPath)
    return clf

if __name__ =="__main__":

    #忽略一些版本不兼容等警告
    warnings.filterwarnings("ignore")

    view_label_index = {"full-shot":0, "whole-body":1, "above-knee":2, "upper-body":3, "lower-body":4, 
                        "upper-cloth":5, "portrait":6, "waist":7, "detail":8, "scene":9}
    direction_label_index = {"left":0, "half-left":1, "center":2, "half-right":3, "right":4, "back":5, "none":6}
    opposite_direction_label_index = {"left":4, "half-left":3, "center":2, "half-right":1, "right":0, "back":5, "none":6}

    labels = read_csv("./final v4.csv")
    # features_ori = read_csv("./frame_data_fix.csv")
    # x_view, x_direction, y_view, y_direction = preprocess_xy(features_ori, labels)
    # print(y_direction)
    # test = pd.Series(y_direction)
    # print(pd.value_counts(test))
    #源数据产生具体看https://blog.csdn.net/ichuzhen/article/details/51768934
    # n_features=57  #每个样本有几个属性或特征
    
    # print("[ INFO ] Now training: VIEW....\n")
    # # _,_, view_clf3 = train(x_view,y_view)
    # view_clf3 = train(x_view,y_view)
    # print("[ INFO ] Now training: DIRECTION....\n")
    # _,_, dirc_clf3 = train(x_direction,y_direction)
    # dirc_clf3 = train(x_direction, y_direction)

    # joblib.dump(view_clf3, 'ml_models/view_clf.m')
    # joblib.dump(dirc_clf3, 'ml_models/direction_clf.m')
    # view_predict('daily_t.csv', 'daily_t_meta.csv')
    # view_predict('dynamic.csv', 'dynamic_meta.csv')
    # view_predict('daily.csv', 'daily_meta.csv')
    # direction_predict('daily_t.csv', 'daily_t_meta.csv')
    # direction_predict('dynamic.csv', 'dynamic_meta.csv')
    # direction_predict('daily.csv', 'daily_meta.csv')