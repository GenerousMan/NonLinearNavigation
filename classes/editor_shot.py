import os
import cv2
import json
import uuid
import skvideo.io
import numpy as np
from functools import total_ordering

@total_ordering
class EditorShot(object):

    def __init__(self, vshot):
        self.id = uuid.uuid4()
        self.template = vshot.tshot.video # shot
        self.video = vshot.video # shot
        # 视频时长 = 获取案例镜头的总帧数/案例镜头的帧率
        self.duration = self.template.frame_count / self.template.fps
        # 提取帧数 = 素材的帧率 * 视频时长
        self.frame_count = int(self.video.fps * self.duration)
        # 起始帧号在cut_shot的start frame里面的frame_number
        self.start = vshot.video.frames[vshot.start].frame_number
        self.end = self.start + self.frame_count
        self.mid = int((self.start + self.end) / 2)
        self.thumbnail_name = str(uuid.uuid4()) + '.jpg'
        self.video_name = str(uuid.uuid4()) + '.mp4'
        self.thumbnail = '/tempoFiles/thumbnails/' + self.thumbnail_name
        self.video_url = '/tempoFiles/videos/' + self.video_name
        self.writer = None

    def set_video_writer(self, compress=False):
        metadata = skvideo.io.ffprobe(self.video.path)
        width = self.video.width
        height = self.video.height
        if compress:
            width = int((width + 1) / 2)
            height = int((height + 1) / 2)
        inputdict = {
            '-r': str(int(self.video.fps))
        }
        outputdict = {
            '-r': str(int(self.video.fps)),
            '-vcodec': 'libx264',
            '-preset': 'ultrafast'
            # '-s': '{}X{}'.format(width, height)
        }
        temp_path = '../api-center/public/tempoFiles/videos/'
        video_path = os.path.join(temp_path, self.video_name)
        self.writer = skvideo.io.FFmpegWriter(video_path, inputdict=inputdict, outputdict=outputdict)

    def release_video_writer(self):
        if self.writer:
            self.writer.close()
            self.writer = None

    def save_thumbnail(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        temp_path = '../api-center/public/tempoFiles/thumbnails/'
        frame = cv2.pyrDown(cv2.pyrDown(frame))
        thumbnail_name = self.thumbnail_name
        thumbnail_path = os.path.join(temp_path, thumbnail_name)
        cv2.imwrite(thumbnail_path, frame)

    def get_video_writer(self, video_path):
        metadata = skvideo.io.ffprobe(self.video.path)
        width = self.video.width
        height = self.video.height
        inputdict = {
            '-r': str(int(self.video.fps))
        }
        outputdict = {
            '-r': str(int(self.video.fps)),
            '-vcodec': 'libx264',
            '-preset': 'ultrafast'
            # '-s': '{}X{}'.format(width, height)
        }
        writer = skvideo.io.FFmpegWriter(video_path, inputdict=inputdict, outputdict=outputdict)
        return writer

    def __eq__(self, other):
        return (self.start == other.start) and (self.end == other.end)

    def __lt__(self, other):
        if self.start == other.start:
            return self.end < other.end
        else:
            return self.start < other.start

    # def generate_temp_files(self):
    #     # 生成封面以及预览视频
    #     self.generate_thumbnail()
    #     self.generate_video()

    # def generate_thumbnail(self):
    #     temp_path = '../api-center/public/tempoFiles/thumbnails/'
    #     mid = int((self.start + self.end) / 2)
    #     cap = cv2.VideoCapture(self.video.path)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    #     _, frame = cap.read()
    #     cap.release()
    #     frame = cv2.pyrDown(cv2.pyrDown(frame))
    #     thumbnail_name = self.thumbnail_name
    #     thumbnail_path = os.path.join(temp_path, thumbnail_name)
    #     cv2.imwrite(thumbnail_path, frame)

    # def generate_video(self, compress=False):
    #     temp_path = '../api-center/public/tempoFiles/videos/'
    #     videogen = skvideo.io.vreader(self.video.path)
    #     # video_name = str(uuid.uuid4()) + '.mp4'
    #     video_name = self.video_name
    #     video_path = os.path.join(temp_path, video_name)
    #     print('Write video:{}'.format(video_name))
    #     writer = self.get_video_writer(video_path, compress=compress)
    #     count = 0
    #     for frame_number, frame in enumerate(videogen):
    #         if self.start <= frame_number < self.end:
    #             count += 1
    #             # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #             if compress:
    #                 frame = cv2.pyrDown(frame)
    #             # writer.write(frame)
    #             writer.writeFrame(frame)
    #         elif frame_number >= self.end:
    #             # writer.close()
    #             # writer.release()
    #             break
    #     writer.close()
    #     print('Finish video:{}, Frame count:{}'.format(video_name, count))

    # def set_video_writer(self, video_path, compress=False):
    #     metadata = skvideo.io.ffprobe(self.video.path)
    #     width = int(metadata["video"]["@width"])
    #     height = int(metadata["video"]["@height"])
    #     if compress:
    #         width = int((width + 1) / 2)
    #         height = int((height + 1) / 2)
    #     inputdict = {
    #         '-r': str(int(self.shot.fps))
    #     }
    #     outputdict = {
    #         '-r': str(int(self.shot.fps)),
    #         '-vcodec': 'libx264',
    #         '-preset': 'ultrafast'
    #         # '-s': '{}X{}'.format(width, height)
    #     }
    #     writer = skvideo.io.FFmpegWriter(video_path, inputdict=inputdict, outputdict=outputdict)
    #     # cap = cv2.VideoCapture(self.shot.file_path)

    #     # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #     # fps = int(self.shot.fps)
    #     # writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    #     # cap.release()
    #     # return writer
    #     return writer