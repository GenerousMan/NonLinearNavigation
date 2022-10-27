
import copy
from classes.vshot import VShot

class ShotGroup(object):

    def __init__(self, tshot, videos):
        self.tshot = tshot
        self.cut_to_vshots(videos)
        self.cut_size()
        self.paths = [None for i in range(len(self.vshots))]

    def cut_to_vshots(self, videos):
        self.vshots = []
        tshot_len = len(self.tshot.video.frames)
        for video in videos:
            count = len(video.frames) - tshot_len + 1
            for i in range(0, count, 3):
                vshot = VShot(video, self.tshot, i, i + tshot_len)
                self.vshots.append(vshot)

    def cut_size(self):
        # 尽可能缩减vshots的量
        new_vshots = []
        window = len(self.tshot.video.frames)
        radius = int(window / 2)
        bools = [True for _ in self.vshots]
        for i, vshot in enumerate(self.vshots):
            if not bools[i]:
                continue
            left = i - radius
            right = i + radius
            left = left if left >= 0 else 0
            right = right if right < len(self.vshots) else (len(self.vshots) - 1)
            temp = []
            for j in range(left, right + 1):
                if self.vshots[j].video.name == vshot.video.name:
                    temp.append(self.vshots[j])
            max_simi = max([vshot.simi for vshot in temp])
            if vshot.simi >= max_simi:
                new_vshots.append(vshot)
            for j in range(left, right + 1):
                if self.vshots[j].video.name == vshot.video.name:
                    if self.vshots[j].simi >= max_simi:
                        bools[j] = True
                    else:
                        bools[j] = False
        new_vshots = sorted(new_vshots,
                    key=lambda vshot: vshot.simi,
                    reverse=True)
        if len(new_vshots) > 50:
            new_vshots = new_vshots[:50]
        self.vshots = new_vshots

    def __str__(self):
        return 'ShotGroup {} has {} vshots'.format(self.tshot.video.name, len(self.vshots))