import os
import pandas as pd
from tools.feature import calc_order_of_tshot

class TShot:

    def __init__(self, video):
        self.video = video
        # 载入预处理的特征信息
        self.load_features()
        # 载入标签，修正镜头的特征值
        self.correct_features()
        # 计算特征权重值
        self.calc_weights()

    def load_features(self):
        self.view = [frame.view for frame in self.video.frames]
        self.direction = [frame.direction for frame in self.video.frames]
        self.pose = [frame.pose for frame in self.video.frames]
        self.motion = [frame.motion for frame in self.video.frames]

    def correct_features(self):
        folder = os.path.split(self.video.path)[0]
        data_path = os.path.join(folder, 'feature_data.csv')
        csv = pd.read_csv(data_path)
        data = csv[csv['file_name']==self.video.name]
        self.view_mark = int(data['view'])
        self.direction_mark = int(data['direction'])
        self.pose_mark = int(data['pose'])
        self.motion_mark = int(data['motion'])
        self.check_pose_feature()
        self.check_motion_feature()

    def check_pose_feature(self):
        up = pd.unique(self.pose)
        if len(up) == 1 and up[0] == 4:
            for index in range(len(self.pose)):
                self.pose[index] = self.pose_mark

    def check_motion_feature(self):
        um = pd.unique(self.motion)
        if len(um) == 1 and um[0] == 3:
            for index in range(len(self.motion)):
                self.motion[index] = self.motion_mark

    def calc_weights(self):
        calc_order_of_tshot(self)

    def __str__(self):
        return str(self.video)