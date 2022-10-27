import time
import skvideo.io
import progressbar
from lib.maskRoi.roi import YOLO_TF
from tools import progress

def calc_roi_of_videos(videos, taskId=None):
    start = time.time()
    total_frame_count = sum([video.frame_count for video in videos])
    temp_frame_count = 0
    yolo = YOLO_TF()
    p = progressbar.ProgressBar()
    p.start(len(videos))
    for vi, video in enumerate(videos):
        p.update(vi+1)
        fps = video.fps
        frame_interval = fps * video.sample_time_interval
        frame_interval = frame_interval if frame_interval >= 1 else 1
        frame_number = 0
        next_frame_number = 0
        frame_count = 0
        cap = skvideo.io.vreader(video.path)
        for frame_number, frame in enumerate(cap):
            if frame_number == int(round(next_frame_number)):
                video.frames[frame_count].roi = yolo.detect_from_cvmat(frame)
                next_frame_number += frame_interval
                frame_count += 1
            if taskId: progress.set_(taskId, 'yolo', temp_frame_count + 1, total_frame_count)
            temp_frame_count += 1
    p.finish()
    yolo.sess.close()
    end = time.time()
    cost = end - start
    print('Roi time cost:', cost)
    print('Frame per second:', total_frame_count / cost)