import time
import skvideo.io
from libs.maskRoi.roi import YOLO_TF

def calc_roi_of_videos(videos, taskId=None):
    start = time.time()
    total_frame_count = sum([video.frame_count for video in videos])
    temp_frame_count = 0
    yolo = YOLO_TF()
    for vi, video in enumerate(videos):
        print(video.name)
        fps = video.fps
        frame_interval = fps * video.sample_time_interval
        frame_interval = frame_interval if frame_interval >= 1 else 1
        # frame_number = 0
        # next_frame_number = 0
        # frame_count = 0
        for i in range(len(video.frames)):
            video.frames[i].roi = yolo.detect_from_cvmat(video.frames[i].frame_np)
        # cap = skvideo.io.vreader(video.path)
        # for frame_number, frame in enumerate(cap):
        #     if frame_number == int(round(next_frame_number)):
        #         print(frame_count)
        #         video.frames[frame_count].roi = yolo.detect_from_cvmat(frame)
        #         next_frame_number += frame_interval
        #         frame_count += 1
    yolo.sess.close()
    end = time.time()
    cost = end - start
    print('Roi time cost:', cost)
    print('Frame per second:', total_frame_count / cost)