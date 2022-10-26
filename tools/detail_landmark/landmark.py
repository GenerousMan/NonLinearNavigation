import os

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
from scipy import spatial
import scipy

from tools.video_features.roi import *
from tools.video_features.alphapose import *

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='constant')

        return img

def write_image(image_now, xy, path = None):
    color = [(255,0,0),(128,0,0),(0,255,0),(0,128,0),(0,0,255),(0,0,128),(255,255,0),(128,128,0)]
    image_now = image_now.astype(np.float32)
    for j in range(xy[0].shape[0]):
        cv2.circle(image_now,(int(xy[0][j][0]),int(xy[0][j][1])),10,color[j],-1)
    if(path!=None):
        cv2.imwrite(path,image_now)
    return image_now


def get_lm_output(output):
    lm_pos_map = output['lm_pos_map']
    batch_size, _, pred_h, pred_w = lm_pos_map.size()
    lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)

    lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped, dim=2).cpu().numpy(), (pred_h, pred_w))
    lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)

    return lm_pos_output

def get_crop_frames(video_test):
    all_frames = []
    with torch.no_grad():
        for i in range(len(video_test.frames)):
            roi = video_test.frames[i].roi
            # print(roi)
            crop_frame = video_test.frames[i].frame_np[roi[0]:roi[2],roi[1]:roi[3]]
            if(crop_frame.shape[0]!=0):
                crop_frame = cv2.resize(crop_frame, (224, 224), interpolation=cv2.INTER_NEAREST)
            else:
                crop_frame = cv2.resize(video_test.frames[i].frame_np, (224, 224), interpolation=cv2.INTER_NEAREST)
            all_frames.append(crop_frame)
    return all_frames


def get_videos_landmark(videos,net):
    for video in videos:
        if(video.has_roi == False):
            calc_roi_of_videos([video])
        all_frames = get_crop_frames(video)
        video_array = np.array(all_frames)
        ld_xy_all = get_frames_landmark(video_array, net)

        # 作相对位置与绝对位置的转换，从裁切处转移至完整画幅
        for i in range(len(video.frames)):
            ld_now = ld_xy_all[i]
            roi = video.frames[i].roi
            if(sum(roi)==0):
                continue
            for j,ld in enumerate(ld_now):
                # print(ld)
                frame = video.frames[i].frame_np
                # print(roi)
                # roi :[y1,x1,y2,x2]
                ld_xy_all[i][j][0] = (roi[3]-roi[1])*ld[0]+roi[1]
                ld_xy_all[i][j][1] = (roi[2]-roi[0])*ld[1]+roi[0]
            write_image(frame,[ld_now],"./test/output/ld_test/test"+str(i)+".png")
        video.lm_seq = ld_xy_all



def get_frames_landmark(video_array, model):
    print(video_array.shape)
    if np.max(video_array)>200:
        video_array= video_array / 255.

    lm_list = []

    video = None
    for i in range(video_array.shape[0]):
        image_array = video_array[i]
        if(image_array.shape[0]==0):
            video = torch.cat([video,image_array],dim = 0)
        image_ori = image_array.copy().astype(np.float32)
        # print(type(image_ori))
        image_ori = cv2.cvtColor(image_ori,cv2.COLOR_RGB2BGR)
        to_tensor = transforms.ToTensor()
        rescale224square = Rescale((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

        image = rescale224square(image_ori)
        image = to_tensor(image)
        
        image = normalize(image)
        image = image.float()
        # print(image.shape)
        image = image.unsqueeze(0)
        if(video == None):
            video = image
        else:
            video = torch.cat([video,image],dim = 0)
    
    print(video.shape)
    batch_size = 1
    batch_num = int(video.shape[0]/batch_size)
    lm_pos_output = []

    for i in range(batch_num):
        #print("batch:",i)
        image = {"image":video[i*batch_size:(i+1)*batch_size]}
        output = model(image)

        output_batch = get_lm_output(output)

        if(len(lm_pos_output)==0):
            lm_pos_output = output_batch
        else:
            lm_pos_output = np.concatenate([lm_pos_output,output_batch],axis = 0)
    
    if(len(lm_pos_output)==0):
        image = {"image":video}
        output = model(image)
        lm_pos_output = get_lm_output(output)

    elif(video_array.shape[0]-batch_size*batch_num >0):
        image = {"image":video[batch_num*batch_size:]}
        output = model(image)
        output_batch = get_lm_output(output)
        lm_pos_output = np.concatenate([lm_pos_output,output_batch],axis = 0)
        
    return lm_pos_output


def get_image_landmark(image_array, model):
    if np.max(image_array)>200:
        image_array= image_array / 255.
    image_ori = image_array.copy()
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    to_tensor = transforms.ToTensor()
    rescale224square = Rescale((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    image = rescale224square(image_ori)
    image = to_tensor(image)
    
    image = normalize(image)
    image = image.float()
    print(image.shape)
    image = image.unsqueeze(0)

    print(image.shape)
    image = {"image":image}

    output = model(image)
    lm_pos_map = output['lm_pos_map']
    batch_size, _, pred_h, pred_w = lm_pos_map.size()
    lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)

    lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped, dim=2).cpu().numpy(), (pred_h, pred_w))
    lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)
    print(lm_pos_output.shape)

    return lm_pos_output

def eucl_dis(xy_1,xy_2):
    return np.sqrt(np.sum(np.square(xy_1-xy_2)))

def verify_neck(lm_xy, pose_xy):
    pose = pose_xy['keypoints'].numpy()
    # print(pose_xy['keypoints'].numpy().shape)
    lshoulder = pose[5]
    rshoulder = pose[6]
    shoulder_dis = eucl_dis(lshoulder, rshoulder)
    lm_lshoulder_dis = eucl_dis(lm_xy,lshoulder)
    lm_rshoulder_dis = eucl_dis(lm_xy,rshoulder)

    sum_lm_shoulder_dis = lm_lshoulder_dis + lm_rshoulder_dis 
    if(abs(sum_lm_shoulder_dis - shoulder_dis) < 0.5*shoulder_dis):
        return lm_xy
    
    else:
        print("not match, change the neck point.")
        return 0.5*(lshoulder+rshoulder)

def verify_sheet(lm_xy,pose_xy,l_r):
    pose = pose_xy['keypoints'].numpy()
    precision = 0.2
    # print(pose_xy)
    lshoulder = pose[5]
    rshoulder = pose[6]
    lelbow = pose[7]
    relbow = pose[8]
    lwrist = pose[9]
    rwrist = pose[10]
    shoulder_dis = eucl_dis(lshoulder, rshoulder)
    if(l_r == 'l'):
        pose_lr = np.array([lshoulder, lelbow, lwrist])
    else:
        pose_lr = np.array([rshoulder, relbow, rwrist])
    max_x = np.max(pose_lr[:,0])
    min_x = np.min(pose_lr[:,0])
    max_y = np.max(pose_lr[:,1])
    min_y = np.min(pose_lr[:,1])

    # print(lm_xy[0], max_x, min_x)
    # print(lm_xy[1], max_y, min_y)
    # print("-------")
    if(lm_xy[0]<max_x+precision*shoulder_dis and lm_xy[0]>min_x-precision*shoulder_dis):
        if(lm_xy[1]<max_y+precision*shoulder_dis and lm_xy[1]>min_y-precision*shoulder_dis):
            # print(lm_xy)
            return lm_xy
    print('not match, change the sheet,',pose_lr[1])
    return pose_lr[1]

def verify_waist(lm_xy,pose_xy,lr):
    pose = pose_xy['keypoints'].numpy()
    precision = 0.7
    lshoulder = pose[5]
    rshoulder = pose[6]
    shoulder_dis = eucl_dis(lshoulder, rshoulder)

    lbutt = pose[11]
    rbutt = pose[12]
    if(lr=='r'):
        butt_x = rbutt[0]
        butt_y = rbutt[1]
    else:
        butt_x = lbutt[0]
        butt_y = lbutt[1]

    if(abs(lm_xy[1]-butt_y)<precision*shoulder_dis):
        return lm_xy
    else:
        return np.array([butt_x,butt_y])

def verify_hem(lm_xy,pose_xy,lr):
    pose = pose_xy['keypoints'].numpy()
    precision = 0.4
    lshoulder = pose[5]
    rshoulder = pose[6]
    shoulder_dis = eucl_dis(lshoulder, rshoulder)
    
    lbutt = pose[11]
    rbutt = pose[12]
    butt_y = 0.5*(lbutt[1]+rbutt[1])

    lknee = pose[13]
    rknee = pose[14]

    if(lr=='r'):
        knee_xy = rknee
    else:
        knee_xy = lknee

    if(lm_xy[1]>butt_y+precision*shoulder_dis):
        return lm_xy
    else:
        return knee_xy

def verify_videos_lm(videos):
    for video in videos:
        if(video.has_pose==False):
            with torch.no_grad():
                calc_pose_of_videos_v2([video])
        for i in range(len(video.frames)):
            # video.frames[i].print_self()
            for j in range(8):
                if(j<=1):
                    video.lm_seq[i][j] = verify_neck(video.lm_seq[i][j], video.frames[i].alphapose)
                if(j==2):
                    video.lm_seq[i][j] = verify_sheet(video.lm_seq[i][j], video.frames[i].alphapose, 'r')
                if(j==3):
                    video.lm_seq[i][j] = verify_sheet(video.lm_seq[i][j], video.frames[i].alphapose, 'l')
                if(j==4):
                    video.lm_seq[i][j] = verify_waist(video.lm_seq[i][j], video.frames[i].alphapose,'r')
                if(j==5):
                    video.lm_seq[i][j] = verify_waist(video.lm_seq[i][j], video.frames[i].alphapose,'l')
                if(j==6):
                    video.lm_seq[i][j] = verify_hem(video.lm_seq[i][j], video.frames[i].alphapose,'r')
                if(j==7):
                    video.lm_seq[i][j] = verify_hem(video.lm_seq[i][j], video.frames[i].alphapose,'l')
        # print(video.lm_seq.shape)
        # for j in range(8):
        #     x = video.lm_seq[:,j,0]
        #     y = video.lm_seq[:,j,1]
        #     x = scipy.signal.savgol_filter(x,3,1)
        #     y = scipy.signal.savgol_filter(y,3,1)
        #     video.lm_seq[:,j,0] = x
        #     video.lm_seq[:,j,1] = y
