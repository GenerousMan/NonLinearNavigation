import os
import pickle
import cv2
import time
import torch
import progressbar
from classes.vshot_v2 import VShot
from model.cut_editor.test_tools import get_test_loader
from tools.cut_editor import load_train_model

class VShotGroup:

    def __init__(self, videos):
        self.videos = videos
        self.vshots = []
        self.calc_vshots_of_video()
        self.cut_scores = []
        self.played_vshots = []
        # self.calc_cut_scores_of_vshots()

    def calc_vshots_of_video(self):
        # shot_lens = [1.5, 2, 3, 4, 5]
        # jumps = [0.75, 1, 1.5, 2, 2.5]
        shot_lens = [1.5, 2, 3, 4]
        jumps = [1.5, 2, 3, 4]
        for video in self.videos:
            ratio = round(1 / video.sample_time_interval)
            vshots = []
            for slen, jump in zip(shot_lens, jumps):
                vs = self.calc_vshots_of_len(video, round(slen * ratio), round(jump * ratio))
                vshots.extend(vs)
            # print(video.name + ":" + str(len(vshots)))
            # self.vshots.append(vshots)
            self.vshots.extend(vshots)
        for vi, vshots in enumerate(self.vshots):
            vshots.vi = vi

    def calc_vshots_of_len(self, video, slen, jump):
        vshots = []
        if len(video.frames) < slen:
            return vshots
        for i in range(0, len(video.frames) - slen + 1, jump):
            vshots.append(VShot(video, i, i + slen))
        return vshots

    def calc_cut_scores_of_vshots(self):
        start = time.time()
        data = []
        for i, ivshot in enumerate(self.vshots):
            _, prev_t = ivshot.get_valid_head_tail_frame()
            for j, jvshot in enumerate(self.vshots):
                curr_h, _ = jvshot.get_valid_head_tail_frame()
                data.append([prev_t.data, curr_h.data, prev_t.flow_row, curr_h.flow_row])
        hsv_data = []
        p = progressbar.ProgressBar()
        p.start(len(data))
        i = 0
        for pdata, cdata, pflow, cflow in data:
            p.update(i+1)
            i += 1
            pdata = cv2.cvtColor(pdata, cv2.COLOR_RGB2HSV)
            cdata = cv2.cvtColor(cdata, cv2.COLOR_RGB2HSV)
            hsv_data.append([pdata, cdata, pflow, cflow])
        p.finish()
        # print(len(hsv_data))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        net = load_train_model()
        net.to(device)
        net.eval()
        net_results = None
        p = progressbar.ProgressBar()
        p.start((len(hsv_data) // 25600) + 1)
        for i in range((len(hsv_data) // 25600) + 1):
            p.update(i+1)
            i_start = i * 25600
            i_end = len(hsv_data) if (i + 1) * 25600 > len(hsv_data) else (i + 1) * 25600
            loader = get_test_loader(hsv_data[i_start:i_end])
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    pdata, cdata, pflow, cflow = batch
                    pdata = pdata.to(device)
                    cdata = cdata.to(device)
                    pflow = pflow.to(device)
                    cflow = cflow.to(device)
                    outputs = net(pdata, cdata, pflow, cflow)
                    results = torch.sigmoid(outputs)
                    if net_results is None:
                        net_results = results
                    else:
                        net_results = torch.cat((net_results, results))
        assert net_results.shape[0] == len(hsv_data)
        results = net_results.view(len(self.vshots), len(self.vshots)).tolist()
        for i, ivshot in enumerate(self.vshots):
            for j, jvshot in enumerate(self.vshots):
                if ivshot == jvshot or ivshot.cross_with(jvshot, 16):
                    results[i][j] = 0.0
        p.finish()
        self.cut_scores = results
        end = time.time()
        print("VShotGroup: cut scores.", len(self.vshots), len(hsv_data), round(end - start, 3))

    def get_playable_percent(self):
        all_frames_count = sum([len(video.frames) for video in self.videos])
        played_frames_count = sum([vshot.len for vshot in self.played_vshots])
        return 1 - (played_frames_count / all_frames_count)

    def reset_all_vshots(self):
        self.played_vshots = []
        for vshot in self.vshots:
            vshot.playable = True

    def reset_farthest_vshot(self):
        if len(self.played_vshots) != 0:
            self.reset_playable_vshots(self.played_vshots[0])
            del self.played_vshots[0]

    def get_playable_vshots(self):
        vshots = [vshot for vshot in self.vshots if vshot.playable]
        if len(vshots) == 0 and len(self.played_vshots) != 0:
            self.reset_farthest_vshot()
            vshots = [vshot for vshot in self.vshots if vshot.playable]
        return vshots

    def set_unplayable_vshots(self, cvshot):
        self.played_vshots.append(cvshot)
        for vshot in self.vshots:
            if vshot.cross_with(cvshot):
                vshot.playable = False
        cvshot.playable = False

    def reset_playable_vshots(self, cvshot):
        for vshot in self.vshots:
            if vshot.cross_with(cvshot):
                vshot.playable = True
        cvshot.playable = True

    def print_vshots_state(self):
        for vshot in self.vshots:
            print(vshot)

    @staticmethod
    def load_vshot_group(folder):
        path = os.path.join(folder, 'vshot_group.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)
        raise IOError('No VShotGroup Data File', folder)

    @staticmethod
    def save_vshot_group(folder, vshot_group):
        # 整体存储
        path = os.path.join(folder, 'vshot_group.pkl')
        f = open(path, 'wb')
        pickle.dump(vshot_group, f)
        f.close()