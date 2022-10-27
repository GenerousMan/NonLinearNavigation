import os
import math
import numpy as np

view_map = {
    0: 0, # long-shot
    1: 1, # full-shot
    2: 1, # full-shot
    3: 2, # mid-shot
    4: 2, # mid-shot
    5: 2, # mid-shot
    6: 3, # close-shot
    7: 3, # close-shot
    8: 3, # close-shot
    9: 5  # none
}

direct_map = {
    0: 0, # left
    1: 1, # half-left
    2: 2, # center
    3: 3, # half-right
    4: 4, # right
    5: 6, # back
    6: 8  # none
}

class VShot:

    def __init__(self, video, start, end):
        self.video = video
        self.start = start
        self.end = end
        self.len = end - start
        self.vi = -1
        self.playable = True
        self.calc_features()

    def calc_features(self):
        frames = [frame for frame in self.video.frames[self.start:self.end]]
        self.view = np.array([view_map[frame.view] for frame in frames])
        self.view_mean = self.view.mean()
        self.direct = np.array([direct_map[frame.direction] for frame in frames])
        self.direct_mean = self.direct.mean()
        # self.pose = [frame.pose for frame in frames]
        # self.motion = [frame.motion for frame in frames]
        self.roi = [frame.roi for frame in frames]
        self.roi_mean = self.calc_smooth_rois().mean()
        self.flow = [frame.flow for frame in frames]
        self.flow_crop = self.calc_flow_crop()
        self.flow_row = [frame.flow_row for frame in frames]
        self.data = [frame.data for frame in frames]

    def view_match(self, view):
        return abs(view - self.view_mean) <= 0.5

    def vshot_view_match(self, pvshot):
        view_diff = np.absolute(self.view - pvshot.view)
        view_diff_mean = view_diff.mean()
        return view_diff_mean

    def direct_match(self, direct):
        return abs(direct - self.direct_mean) <= 1.5
        # return abs(direct - self.direct_mean) <= 0.5

    def vshot_direct_match(self, pvshot):
        direct_diff = np.absolute(self.direct - pvshot.direct)
        direct_diff_mean = direct_diff.mean()
        return direct_diff_mean

    def calc_smooth_rois(self):
        w = self.video.width
        rois = np.array([(roi[1]+roi[3])/2/w for roi in self.roi])
        rois = np.where(rois > 0.0, rois, 0.5)
        N = 7
        hN = N // 2
        weights = np.hanning(N)
        smooth_rois = np.convolve(weights/weights.sum(), rois, mode='full')[hN:-hN]
        smooth_rois[0:hN] = rois[0:hN]
        smooth_rois[-hN:] = rois[-hN:]
        return smooth_rois

    def calc_flow_crop(self):
        flow_crop = []
        for flow in self.flow:
            mid = math.floor(self.roi_mean * flow.shape[1])
            if flow.shape[1] <= 11:
                flow_crop.append((mid, flow))
                continue
            if (mid - 5) < 0:
                start = 0
                end = start + 11
            elif (mid + 5) >= flow.shape[1]:
                end = flow.shape[1]
                start = end - 11
                mid = mid - start
            else:
                start = mid - 5
                end = mid + 5 + 1
                mid = mid - start
            flow_crop.append((mid, flow[:, start:end]))
        return flow_crop

    def calc_flow_diff(self, pvshot):
        diff = 0.0
        for (v1_mid, v1_flow), (v2_mid, v2_flow) in zip(self.flow_crop, pvshot.flow_crop):
            # v1_mid, v1_flow = VShot.calc_frame_flow(frame1, mroi_1)
            # v2_mid, v2_flow = VShot.calc_frame_flow(frame2, mroi_2)
        #     print(v1_mid, v1_flow.shape)
        #     print(v2_mid, v2_flow.shape)
            if v1_flow.shape[1] < v2_flow.shape[1]:
                radius = v1_flow.shape[1] // 2
                if (v2_mid - radius) < 0:
                    v2_start = 0
                    v2_end = v2_start + v1_flow.shape[1]
                elif (v2_mid + radius) >= v2_flow.shape[1]:
                    v2_end = v2_flow.shape[1]
                    v2_start = v2_end - v1_flow.shape[1]
                else:
                    v2_start = v2_mid - radius
                    v2_end = v2_start + v1_flow.shape[1]
                v2_flow = v2_flow[:, v2_start:v2_end]
        #         print("New v2_flow:", v2_start, v2_end)
            elif v1_flow.shape[1] > v2_flow.shape[1]:
                radius = v2_flow.shape[1] // 2
                if (v1_mid - radius) < 0:
                    v1_start = 0
                    v1_end = v1_start + v2_flow.shape[1]
                elif (v1_mid + radius) >= v1_flow.shape[1]:
                    v1_end = v1_flow.shape[1]
                    v1_start = v1_end - v2_flow.shape[1]
                else:
                    v1_start = v1_mid - radius
                    v1_end = v1_start + v2_flow.shape[1]
                v1_flow = v1_flow[:, v1_start:v1_end]
        #         print("New v1_flow:", v1_start, v1_end)
            x_diff = v1_flow[:,:,0] - v2_flow[:,:,0]
            y_diff = v1_flow[:,:,1] - v2_flow[:,:,1]
            diff += np.mean(np.sqrt((x_diff * x_diff + y_diff * y_diff)))
        return diff

    @staticmethod
    def calc_flow_diff_of_pair(vshots):
        diff = 0.0
        for i, ivshot in enumerate(vshots):
            for j, jvshot in enumerate(vshots[i+1:]):
                diff += ivshot.calc_flow_diff(jvshot)
        return diff

    def get_valid_head_tail_frame(self):
        frames = self.video.frames[self.start:self.end]
        if len(frames) == 1:
            return (None, None)
        elif len(frames) == 2:
            return (frames[1], frames[1])
        elif len(frames) <= 8:
            mid = len(frames) // 2
            return (frames[mid], frames[mid])
        else:
            return (frames[3], frames[-4])

    def cross_with(self, other, gap=0):
        if self.video != other.video:
            return False
        else:
            if self.end + gap < other.start or self.start - gap > other.end:
                return False
            else:
                return True

    def __eq__(self, other):
        video_eq = self.video == other.video
        start_eq = self.start == other.start
        end_eq = self.end == other.end
        return video_eq and start_eq and end_eq

    def __str__(self):
        name = self.video.name
        video_frame_count = self.video.frame_count
        sti = self.video.sample_time_interval
        video_len = round(video_frame_count / self.video.fps)
        vshot_len = self.len * sti
        vshot_start = self.start * sti
        vshot_end = self.end * sti
        return 'VShot {} {} {} {}-{} {} {}'.format(name, video_len, vshot_len, vshot_start, vshot_end, round(self.view_mean, 3), round(self.direct_mean, 3))