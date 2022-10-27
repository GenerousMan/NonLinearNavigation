import os
import numpy as np

class ShotLine(object):

    def __init__(self, vshots):
        self.vshots = vshots
        self.calc_simi()

    def calc_simi(self):
        dists = []
        for vshot in self.vshots:
            dists.append(vshot.view_dist)
        self.view_simi = int((1 - np.mean(dists)) * 100)
        dists = []
        for vshot in self.vshots:
            dists.append(vshot.direction_dist)
        self.direction_simi = int((1 - np.mean(dists)) * 100)
        dists = []
        for vshot in self.vshots:
            dists.append(vshot.pose_dist)
        self.pose_simi = int((1 - np.mean(dists)) * 100)
        dists = []
        for vshot in self.vshots:
            dists.append(vshot.motion_dist)
        self.motion_simi = int((1 - np.mean(dists)) * 100)
        simis = []
        for vshot in self.vshots:
            simis.append(vshot.simi)
        self.simi = int((np.mean(simis)) * 100)

    def saveThumbnails(self):
        print('saveThumbnails')
        for vshot in self.vshots:
            vshot.getThumbnail()

    def getThumbnails(self):
        print('getThumbnails')
        thumbnails = []
        for vshot in self.vshots:
            thumbnails.append(vshot.thumbnail)
        return thumbnails

    def getTemplateThumbnails(self):
        print('getTemplateThumbnails')
        real_path = '../api-center/public/videos/' + self.template_path + '/'
        url_path = '/videos/' + self.template_path + '/'
        thumbnails = os.listdir(real_path)
        thumbnails = [file for file in thumbnails if file.endswith('.jpg')]
        thumbnails = sorted(thumbnails, key=lambda t: int(t[:-4]))
        thumbnails = [url_path + file for file in thumbnails]
        return thumbnails

    def get_match_data(self):
        data = {
            'template': self.template_path,
            'template_video': '/videos/' + self.template_path + '/video.mp4',
            'result_video': '/tempoFiles/videos/' + self.result_name,
            'template_thumbnails': self.getTemplateThumbnails(),
            'result_thumbnails': self.getThumbnails(),
            'simi': self.simi,
            'view_simi': self.view_simi,
            'direction_simi': self.direction_simi,
            'pose_simi': self.pose_simi,
            'motion_simi': self.motion_simi
        }
        return data

    def __str__(self):
        names = [vshot.video.name for vshot in self.vshots]
        return str(names)