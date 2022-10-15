from __future__ import division
import argparse
import os
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import AttrPredictor, CatePredictor
from mmfashion.models import build_predictor
from mmfashion.utils import get_img_tensor

import numpy as np

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

if __name__ == "__main__":
    imgs_path = "/root/ali-2021/Git_project/Fasion/demo/imgs/test_cate/"

    imgs = os.listdir(imgs_path)
    imgs.sort()
    print(imgs)
    checkpoint = "checkpoints/category_att_pred/global-vgg.pth"
    config_path = "configs/category_attribute_predict/global_predictor_vgg.py"
    cfg = Config.fromfile(config_path)
    use_cuda = True
    cate_probs = []
    attr_probs = []
    for img_path in imgs:
        img_tensor = get_img_tensor(imgs_path+img_path, use_cuda)
        # global attribute predictor will not use landmarks
        # just set a default value
        landmark_tensor = torch.zeros(8)

        model = build_predictor(cfg.model)
        load_checkpoint(model, checkpoint)
        print('model loaded from {}'.format(checkpoint))
        if use_cuda:
            model.cuda()
            landmark_tensor = landmark_tensor.cuda()

        model.eval()

        # predict probabilities for each attribute
        attr_prob, cate_prob = model(
            img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)
        attr_probs.append(attr_prob.cpu())
        cate_probs.append(cate_prob.cpu())

        # print(attr_prob)
        # print(cate_prob)
        attr_predictor = AttrPredictor(cfg.data.test)
        cate_predictor = CatePredictor(cfg.data.test)

        # attr_predictor.show_prediction(attr_prob)
        # cate_predictor.show_prediction(cate_prob)

    attr_distance = calc_distance_matrix(attr_probs)
    cate_distance = calc_distance_matrix(cate_probs)

    print_distance("attr distance : ", attr_distance)
    print_distance("cate distance : ", cate_distance)
