import os
import pickle
import numpy as np
import pandas as pd
import analyze as aly

class TShotBag(object):

    def __init__(self, folder):
        tshots = aly.load_shots(folder)
        self.folder = folder
        self.calc_view_hist(tshots)
        self.calc_direction_hist(tshots)
        self.calc_pose_hist(tshots)
        self.calc_motion_hist(tshots)

    def calc_view_hist(self, tshots):
        views = []
        for ts in tshots:
            ts_views = [frame.view for frame in ts.shot.frames]
            views.extend(ts_views)
        default_hist = pd.Series([0 for i in range(10)])
        t_view_hist = pd.value_counts(views)
        self.view_hist = default_hist.add(t_view_hist, fill_value=0)
        self.view_hist = self.view_hist / self.view_hist.sum()
        # print(self.view_hist)

    def calc_direction_hist(self, tshots):
        directs = []
        for ts in tshots:
            ts_directs = [frame.direction for frame in ts.shot.frames]
            directs.extend(ts_directs)
        default_hist = pd.Series([0 for i in range(7)])
        t_direct_hist = pd.value_counts(directs)
        self.direct_hist = default_hist.add(t_direct_hist, fill_value=0)
        self.direct_hist = self.direct_hist / self.direct_hist.sum()
        # print(self.direct_hist)

    def calc_pose_hist(self, tshots):
        poses = []
        for ts in tshots:
            ts_poses = [frame.pose for frame in ts.shot.frames]
            poses.extend(ts_poses)
        default_hist = pd.Series([0 for i in range(5)])
        t_pose_hist = pd.value_counts(poses)
        self.pose_hist = default_hist.add(t_pose_hist, fill_value=0)
        self.pose_hist = self.pose_hist / self.pose_hist.sum()
        # print(self.pose_hist)

    def calc_motion_hist(self, tshots):
        motions = []
        for ts in tshots:
            ts_motions = [frame.motion for frame in ts.shot.frames]
            motions.extend(ts_motions)
        default_hist = pd.Series([0 for i in range(4)])
        t_motion_hist = pd.value_counts(motions)
        self.motion_hist = default_hist.add(t_motion_hist, fill_value=0)
        self.motion_hist = self.motion_hist / self.motion_hist.sum()

    @staticmethod
    def hist_distance(hist1, hist2):
        # 计算两个直方图的Hellinger Distance
        mult = hist1.multiply(hist2)
        bc = np.sqrt(mult).sum()
        dis = np.sqrt(np.clip(1 - bc, 0, 1))
        return dis

    @staticmethod
    def save(tsb):
        path = os.path.join(tsb.folder, 'tshot_bag.pkl')
        f = open(path, 'wb')
        pickle.dump(tsb, f)
        f.close()

    @staticmethod
    def load(folder):
        path = os.path.join(folder, 'tshot_bag.pkl')
        with open(path, 'rb') as f:
            tsb = pickle.load(f)
            return tsb
        return None