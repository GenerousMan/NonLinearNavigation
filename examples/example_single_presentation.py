import os, sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
import torch
from skimage import io, transform
from torchvision import transforms
from skimage import io, transform
from torch.utils.data import DataLoader
import torch.nn.functional as F
import shutil

import libs.Landmark.utils
from libs.Landmark.models import Network
from libs.Landmark.utils import cal_loss, Evaluator
from libs.Detail.ResnetFinetune import TransferModel
from classes.video import Video
from tools.video_features.alphapose import calc_pose_of_videos_v2
from tools.video_features.roi import calc_roi_of_videos
from tools.video_features.combine_pose_roi import combine_roi_pose, calc_features
from tools.detail_landmark.detail import get_image_details, get_video_details
from tools.detail_landmark.landmark import *
from tools.video_features.roi import *
from tools.video_features.alphapose import calc_pose_of_videos_v2
from tools.video_features.roi import calc_roi_of_videos
from tools.video_features.combine_pose_roi import combine_roi_pose, calc_features
from tools.yaml_tool import *

def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "/" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

detail_labels = ['hem','neck','sheet','waist']
detail_thres = [0.3,0.3,0.3,0.3]
# detail_labels = ['hem','low','neck','None','sheet','up','waist']
# landmark
net = torch.nn.DataParallel(Network(dataset=['fld'], flag=True)).cuda()
weights = torch.load('./libs/Landmark/models/model_07.pkl')
# weights = utils.load_weight(net, weights)
net.load_state_dict(weights)

# detail
model = TransferModel(n_class = 4).cuda()
model.load_state_dict(torch.load('libs/Detail/models/4_38_model.pkl'))
model.eval()

# load videos and preprocess. 
video_dir = './test/cuts/8/'
video_names = os.listdir(video_dir)

output_dir = './test/output/'+video_dir.split('test/')[-1]
if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)
else:
    del_file(output_dir)

video_obs = [Video((video_dir + video)) for video in video_names if video.split(".")[-1] == "mp4"]
calc_pose_of_videos_v2(video_obs)
calc_roi_of_videos(video_obs)
combine_roi_pose(video_obs)
calc_features(video_obs)

full_shots = []
detail_shots = []

for i,video in enumerate(video_obs):
    print(video.name,video.frames[0].view,video.frames[0].roi)
    if(video.frames[0].view<=2 and video.frames[0].roi!=[0,0,0,0]):
        full_shots.append(video)
    else:
        detail_shots.append(video)

    # video_test = Video(video_dir + video)
    # calc_roi_of_videos([video_test])
    # print(video_test.frames[0].roi)
    # roi = video_test.frames[0].roi
    # crop_frame = video_test.frames[0].frame_np[roi[0]:roi[2],roi[1]:roi[3]]
    # print(crop_frame.shape)

print('full:',[full_shots[i].name for i in range(len(full_shots))])
full_save_dir = output_dir+'full/'
if(not os.path.exists(full_save_dir)):
    os.makedirs(full_save_dir)
for f_shot in full_shots:
    shutil.copy(video_dir+f_shot.name, full_save_dir+f_shot.name)
    video_name = full_save_dir+f_shot.name
    video_full = Video(video_name)
    calc_pose_of_videos_v2([video_full])
    calc_roi_of_videos([video_full])
    

    net = torch.nn.DataParallel(Network(dataset=['fld'], flag=True)).cuda()
    weights = torch.load('./libs/Landmark/models/model_07.pkl')
    # weights = utils.load_weight(net, weights)
    net.load_state_dict(weights)

    get_videos_landmark([video_full],net)
    verify_videos_lm([video_full])
    full_keypoints = video_full.lm_seq
    full_name = f_shot.name
    duration = int(video_full.sample_time_interval * 1000)

useful_keypoints = [[],[],[],[]]
neck_shot = None
sheet_shot = None
waist_shot = None
hem_shot = None

print('detail:',[detail_shots[i].name for i in range(len(detail_shots))])
with torch.no_grad():
    for i,d_shot in enumerate(detail_shots):
        print(d_shot.name)
        d_shot_array = np.array([d_shot.frames[i].frame_np for i in range(len(d_shot.frames))])
        result_shot = (F.softmax(torch.mean(get_video_details(d_shot_array,model),axis = 0))).cpu().detach().numpy()
        print(result_shot,detail_labels[np.argmax(result_shot)])
        video_save_dir = output_dir+detail_labels[np.argmax(result_shot)]+'/'
        if(np.max(result_shot)>detail_thres[np.argmax(result_shot)]):
            if(not os.path.exists(video_save_dir)):
                os.makedirs(video_save_dir)
            shutil.copy(video_dir+d_shot.name, video_save_dir+d_shot.name)

            if(detail_labels[np.argmax(result_shot)] == 'neck'):
                useful_keypoints[0] = full_keypoints[:,0]
                neck_shot = d_shot.name
            if(detail_labels[np.argmax(result_shot)] == 'sheet'):
                useful_keypoints[1] = full_keypoints[:,3]
                sheet_shot = d_shot.name
            if(detail_labels[np.argmax(result_shot)] == 'waist'):
                useful_keypoints[2] = full_keypoints[:,5]
                waist_shot = d_shot.name
            if(detail_labels[np.argmax(result_shot)] == 'hem'):
                useful_keypoints[3] = full_keypoints[:,6]
                hem_shot = d_shot.name
detail_shot_list = [neck_shot, sheet_shot, waist_shot, hem_shot]
flow = Aniv_yaml(full_name, detail_shot_list, useful_keypoints, duration)

print(np.array(useful_keypoints).shape, np.array(full_keypoints).shape)



# 保存文件格式的实验数据
exp_path = "./exp/"+video_dir.split("test/")[-1]
useful_keypoints = np.array([useful_keypoints[i] for i in range(len(useful_keypoints)) if useful_keypoints[i]!=[]])
detail_shot_list = [detail_shot_list[i] for i in range(len(detail_shot_list)) if detail_shot_list[i]!=None]
# 原始的图像帧
ori_image_path = exp_path+"image_ori/"
if(not os.path.exists(ori_image_path)):
    os.makedirs(ori_image_path)
ori_frames = video_full.frames
for i,frame in enumerate(ori_frames):
    cv2.imwrite(ori_image_path+ str(i)+".png",frame.frame_np)

# 每一帧的关键点打标结果
vis_image_path = exp_path+"image_vis/"
if(not os.path.exists(vis_image_path)):
    os.makedirs(vis_image_path)
print(useful_keypoints.shape)
for i,frame in enumerate(ori_frames):
    write_image(frame.frame_np,np.array([useful_keypoints[:,i]]) , path = vis_image_path+ str(i)+".png")

# 特写镜头
close_shot_path = exp_path+"close_shot/"
if(not os.path.exists(close_shot_path)):
    os.makedirs(close_shot_path)
for detail in detail_shot_list:
    shutil.copy(video_dir+detail, close_shot_path+detail)
# print(flow)

# 保存yaml格式的结果
with open(video_dir+'yaml_output.yaml', "w", encoding="utf-8") as f:
    yaml.dump(flow, f)
