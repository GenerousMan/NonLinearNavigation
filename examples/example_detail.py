import os, sys

sys.path.append(os.getcwd())
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import time
import cv2
from torchvision import models
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
torch.cuda.set_device(0)

from libs.Detail.ResnetFinetune import TransferModel
from tools.detail_landmark.detail import get_image_details, get_video_details


model = TransferModel().cuda()
model.load_state_dict(torch.load('libs/Detail/4_model.pkl'))

model.eval()

video_name = './xxx.mp4' # video to be predicted.

cap = cv2.VideoCapture(video_name)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
fourcc = cv2.VideoWriter_fourcc(*'H264')

all_frames = []
while (cap.isOpened()):
    ret, frame = cap.read()
    if(ret == True):
        all_frames.append(frame)
        # print(frame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cap.release()
        break

video_array = np.array(all_frames)
print(get_video_details(video_array,model))

names = ['Hem','Neckline','Sheet','Waist']
scores = (100*F.softmax(get_image_details(video_array[0],model)).cpu().detach().numpy())
scores = np.around(scores,2)
for i in range(scores.shape[1]):
    print(names[i],":",scores[0][i])
# print(scores)