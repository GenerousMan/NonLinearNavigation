import os
import cv2
import uuid

class VShot:

    def __init__(self, video, tshot, start, end):
        self.video = video
        self.start = start
        self.end = end
        self.tshot = tshot
        self.calc_features()
        self.calc_simi()

    def calc_features(self):
        frames = [self.video.frames[i] for i in range(self.start, self.end)]
        self.view = [frame.view for frame in frames]
        self.direction = [frame.direction for frame in frames]
        self.pose = [frame.pose for frame in frames]
        self.motion = [frame.motion for frame in frames]

    def calc_simi(self):
        self.calc_view_dist()
        self.calc_pose_dist()
        self.calc_direction_dist()
        self.calc_motion_dist()
        view_dist = self.tshot.view_weight * self.view_dist
        direction_dist = self.tshot.direction_weight * self.direction_dist
        pose_dist = self.tshot.pose_weight * self.pose_dist
        motion_dist = self.tshot.motion_weight * self.motion_dist
        self.simi = 1 - (view_dist + direction_dist + pose_dist + motion_dist)

    def cross_with(self, other):
        if self.video != other.video:
            return False
        else:
            if self.end < other.start or self.start > other.end:
                return False
            else:
                return True

    def __eq__(self, other):
        video_eq = self.video == other.video
        start_eq = self.start == other.start
        end_eq = self.end == other.end
        return video_eq and start_eq and end_eq

    def getThumbnail(self):
        '''
            在临时文件夹'../tempoFiles/thumbnails/'下生成该cutshot首帧的图像作为封面
        '''
        temp_path = '../api-center/public/tempoFiles/thumbnails/'
        cap = cv2.VideoCapture(self.video.path)
        if cap.isOpened():
            start_frame_number = self.video.frames[self.start].frame_number
            end_frame_number = self.video.frames[self.end - 1].frame_number
            frame_number = int((start_frame_number + end_frame_number) / 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            _, frame = cap.read()
            frame = cv2.pyrDown(cv2.pyrDown(frame))
            if not _: raise Exception('Fail to read from{}'.format(self.video.path))
            thumbnail_name = str(uuid.uuid4()) + '.jpg'
            thumbnail_path = os.path.join(temp_path, thumbnail_name)
            cv2.imwrite(thumbnail_path, frame)
            self.thumbnail = '/tempoFiles/thumbnails/' + thumbnail_name
        cap.release()

    def calc_view_dist(self):
        distance = []
        for tag1, tag2 in zip(self.tshot.view, self.view):
            distance.append(VShot.view_tag_distance(tag1, tag2))
        self.view_dist = sum(distance) / len(distance)

    def calc_direction_dist(self):
        distance = []
        for tag1, tag2 in zip(self.tshot.direction, self.direction):
            distance.append(VShot.direction_tag_distance(tag1, tag2))
        self.direction_dist = sum(distance) / len(distance)

    def calc_pose_dist(self):
        distance = []
        for tag1, tag2 in zip(self.tshot.pose, self.pose):
            distance.append(VShot.pose_tag_distance(tag1, tag2))
        self.pose_dist = sum(distance) / len(distance)

    def calc_motion_dist(self):
        distance = []
        for tag1, tag2 in zip(self.tshot.motion, self.motion):
            distance.append(VShot.motion_tag_distance(tag1, tag2))
        self.motion_dist = sum(distance) / len(distance)

    @staticmethod
    def view_tag_distance(tag1, tag2):
        # {"full-shot":0, "whole-body":1, "above-knee":2, "upper-body":3, "lower-body":4,
        # "upper-cloth":5, "portrait":6, "waist":7, "detail":8, "scene":9}
        distance = [
            [0, 1, 2, 3, 2, 4, 4, 3, 5, 6],
            [1, 0, 1, 2, 1, 3, 3, 2, 4, 6],
            [2, 1, 0, 1, 2, 2, 2, 1, 3, 6],
            [3, 2, 1, 0, 3, 1, 1, 2, 2, 6],
            [2, 1, 2, 3, 0, 4, 4, 3, 1, 6],
            [4, 3, 2, 1, 4, 0, 2, 3, 1, 6],
            [4, 3, 2, 1, 4, 2, 0, 3, 1, 6],
            [3, 2, 1, 2, 3, 3, 3, 0, 1, 6],
            [5, 4, 3, 2, 1, 1, 1, 1, 0, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 0]
        ]
        max_disance = 6
        return distance[tag1][tag2] / max_disance

    @staticmethod
    def direction_tag_distance(tag1, tag2):
        # {"left":0, "half-left":1, "center":2, "half-right":3, "right":4, "back":5, "none":6}
        distance = [
            [0, 1, 2, 3, 4, 1, 5],
            [1, 0, 1, 2, 3, 2, 5],
            [2, 1, 0, 1, 2, 3, 5],
            [3, 2, 1, 0, 1, 2, 5],
            [4, 3, 2, 1, 0, 1, 5],
            [1, 2, 3, 2, 1, 0, 5],
            [5, 5, 5, 5, 5, 5, 0]
        ]
        max_disance = 5
        return distance[tag1][tag2] / max_disance

    @staticmethod
    def pose_tag_distance(tag1, tag2):
        # {'stand': 0,'sit': 1,'walk': 2,'spin': 3,'none': 4 }
        distance = [
            [0, 2, 1, 1, 3],
            [2, 0, 2, 2, 3],
            [1, 2, 0, 1, 3],
            [1, 2, 1, 0, 3],
            [3, 3, 3, 3, 0]
        ]
        max_disance = 3
        return distance[tag1][tag2] / max_disance

    @staticmethod
    def motion_tag_distance(tag1, tag2):
        # {'still': 0,'low': 1,'high': 2,'none': 3}
        distance = [
            [0, 1, 2, 3],
            [1, 0, 1, 3],
            [2, 1, 0, 3],
            [3, 3, 3, 0],
        ]
        max_disance = 3
        return distance[tag1][tag2] / max_disance