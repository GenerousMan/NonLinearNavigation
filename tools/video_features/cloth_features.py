from __future__ import division
import argparse
import os
import sys
sys.path.append(os.getcwd())
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import AttrPredictor, CatePredictor
from mmfashion.models import build_predictor
from mmfashion.utils import get_img_tensor
import cv2
from PIL import Image, ImageDraw

from classes.video import Video

def PIL_tensor_image(PIL_image):
    original_w, original_h = PIL_image.size

    img_size = (224, 224)  # crop image to (224, 224)
    PIL_image.thumbnail(img_size, Image.ANTIALIAS)
    img = PIL_image.convert('RGB')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    img_tensor = transform(img)
    return img_tensor

def calc_cloth_of_videos(videos,use_cuda=True):
    torch.cuda.empty_cache()
    checkpoint = "/root/ali-2021/mm_sim/libs/Fasion/checkpoints/category_att_pred/global-vgg.pth"
    config_path = "/root/ali-2021/mm_sim/libs/Fasion/configs/category_attribute_predict/global_predictor_vgg.py"
    cfg = Config.fromfile(config_path)
    use_cuda = True
    batch_size = 1
    model = build_predictor(cfg.model)
    load_checkpoint(model, checkpoint)
    print("[ INFO ] Calculating clothes' attribute and category")
    for video in videos:
        print(video.name)
        imgs_numpy = [video.frames[i].frame_np for i in range(len(video.frames))]
        imgs_PIL = [Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) for img in imgs_numpy]
        imgs_tensor = torch.stack([PIL_tensor_image(img_PIL) for img_PIL in imgs_PIL], dim = 0)
        # print(imgs_tensor.shape)
        landmark_tensor = torch.zeros(8)


        # print('model loaded from {}'.format(checkpoint))
        if use_cuda:
            model.cuda()
            landmark_tensor = landmark_tensor.cuda()

        model.eval()
        batch_num = imgs_tensor.shape[0]//batch_size
        # print(batch_num,imgs_tensor.shape[0])
        # predict probabilities for each attribute
        batch_data = imgs_tensor[:batch_size]

        if use_cuda:
            batch_data = batch_data.cuda()

        attr_prob, cate_prob = model(
            batch_data, attr=None, landmark=landmark_tensor, return_loss=False)
            
        for i in range(1,batch_num):
            # print(i,batch_num)
            batch_data = imgs_tensor[i*batch_size:(i+1)*batch_size]
            if use_cuda:
                batch_data = batch_data.cuda()
            attr_prob_batch, cate_prob_batch = model(
            batch_data, attr=None, landmark=landmark_tensor, return_loss=False)
            # print(attr_prob_batch.shape,attr_prob.shape)
            attr_prob = torch.cat([attr_prob, attr_prob_batch],dim = 0)
            cate_prob = torch.cat([cate_prob, cate_prob_batch],dim = 0)
            torch.cuda.empty_cache()
        # 批处理，避免显存不够用。
        if(imgs_tensor.shape[0]-batch_num*batch_size!=0):
            batch_data = imgs_tensor[batch_num*batch_size:]
            if use_cuda:
                batch_data = batch_data.cuda()
            attr_prob_batch, cate_prob_batch = model(
            batch_data, attr=None, landmark=landmark_tensor, return_loss=False)
            # print(attr_prob_batch.shape,attr_prob.shape)
            attr_prob = torch.cat([attr_prob, attr_prob_batch],dim=0)
            cate_prob = torch.cat([cate_prob, cate_prob_batch],dim=0)
        for i, frame in enumerate(video.frames):
            frame.cloth_cate = cate_prob[i].cpu().detach().numpy()
            frame.cloth_attr = attr_prob[i].cpu().detach().numpy()
        torch.cuda.empty_cache()


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return round(dist[0][0],3)

def calc_distance_matrix(input_list):
    input_list = np.array(input_list)
    matrix_all = []
    for a in input_list:
        matrix_line = []
        for b in input_list:
            matrix_line.append(cosine_distance(a.detach().numpy(),b.detach().numpy()))
        matrix_all.append(matrix_line)

    return matrix_all

def print_distance(str_output, distance_list):
    print(str_output)
    for distance in distance_list:
        print(distance)

def get_image_cloth_feature(img_path):
    pass

if __name__ == "__main__":

    calc_cloth_of_videos([Video("./source/video/1_4.mp4")])


    # imgs_path = "/root/ali-2021/Git_project/Fasion/demo/imgs/test_cate/"

    # imgs = os.listdir(imgs_path)
    # imgs.sort()
    # print(imgs)
    # checkpoint = "/root/ali-2021/mm_sim/libs/Fasion/checkpoints/category_att_pred/global-vgg.pth"
    # config_path = "/root/ali-2021/mm_sim/libs/Fasion/configs/category_attribute_predict/global_predictor_vgg.py"
    # cfg = Config.fromfile(config_path)
    # use_cuda = True
    # cate_probs = []
    # attr_probs = []
    # for img_path in imgs:
    #     img_tensor = get_img_tensor(imgs_path+img_path, use_cuda)
    #     # global attribute predictor will not use landmarks
    #     # just set a default value
    #     landmark_tensor = torch.zeros(8)

    #     model = build_predictor(cfg.model)
    #     load_checkpoint(model, checkpoint)
    #     print('model loaded from {}'.format(checkpoint))
    #     if use_cuda:
    #         model.cuda()
    #         landmark_tensor = landmark_tensor.cuda()

    #     model.eval()

    #     # predict probabilities for each attribute
    #     attr_prob, cate_prob = model(
    #         img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)
    #     attr_probs.append(attr_prob.cpu())
    #     cate_probs.append(cate_prob.cpu())

    #     # print(attr_prob)
    #     # print(cate_prob)
    #     attr_predictor = AttrPredictor(cfg.data.test)
    #     cate_predictor = CatePredictor(cfg.data.test)

    #     # attr_predictor.show_prediction(attr_prob)
    #     # cate_predictor.show_prediction(cate_prob)

    # attr_distance = calc_distance_matrix(attr_probs)
    # cate_distance = calc_distance_matrix(cate_probs)

    # print_distance("attr distance : ", attr_distance)
    # print_distance("cate distance : ", cate_distance)
