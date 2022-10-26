import os, sys

sys.path.append(os.getcwd())

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from libs.Landmark.models import Network
from libs.Landmark.utils import cal_loss, Evaluator
import libs.Landmark.utils
from skimage import io, transform
from torchvision import transforms
import cv2
from skimage import io, transform

from tools.detail_landmark.landmark import *
from tools.video_features.roi import *
from tools.video_features.alphapose import *
from classes.video import Video

def write_video_lm(video_name,video_ob):
    cap = cv2.VideoCapture(video_name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('./test/output/ld_test/'+video_ob.name.split(".")[0]+'.avi', fourcc, fps, (width, height))
    frame_count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        # print(frame.shape)
        if(ret == True):
            # print(frame.shape)
            for j,xy in enumerate(video_ob.lm_seq):
                # 找到帧数大于当前帧的关键帧，并将其与之前的关键帧相减作差值
                if(video_ob.frames[j].frame_number>frame_count):
                    break
            xy_diff = video_ob.lm_seq[j] - video_ob.lm_seq[j-1]
            now_xy = video_ob.lm_seq[j-1] + xy_diff *\
                    (frame_count-video_ob.frames[j-1].frame_number)/(video_ob.frames[j].frame_number - video_ob.frames[j-1].frame_number)
            # print('before:',video_ob.lm_seq[j-1])
            # print('after:',video_ob.lm_seq[j])
            # print('now:',now_xy)
            # print('-------------')
            lm_frame_now = write_image(frame, [now_xy], "./test/output/ld_test/test_lm_frame.png")
            
            #print(lm_frame_now)
            out.write(lm_frame_now.astype(np.uint8))
            frame_count+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()
            out.release()
            break

video_name = './test/alitest/full-2.mp4'
with torch.no_grad():
    video_test = Video(video_name)
    calc_pose_of_videos_v2([video_test])
    calc_roi_of_videos([video_test])
    

    net = torch.nn.DataParallel(Network(dataset=['fld'], flag=True)).cuda()
    weights = torch.load('./libs/Landmark/models/model_07.pkl')
    # weights = utils.load_weight(net, weights)
    net.load_state_dict(weights)


    get_videos_landmark([video_test],net)
    verify_videos_lm([video_test])

    write_video_lm(video_name, video_test)
    print(video_test.lm_seq.shape)