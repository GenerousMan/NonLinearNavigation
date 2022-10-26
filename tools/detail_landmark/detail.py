import os
import cv2
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import time
from torchvision import models
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from skimage import io, transform

torch.cuda.set_device(0)

class TransferModel(nn.Module):
    def __init__(self,
                base_model : str = 'resnet50',
                pretrain : bool = True,
                n_class : int = 4):
        super(TransferModel, self).__init__()
        self.base_model = base_model
        self.pretrain = pretrain
        self.n_class = n_class
        if self.base_model == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
            n_features = self.model.fc.in_features
            fc = torch.nn.Linear(n_features, n_class)
            self.model.fc = fc
        else:
            # Use other models you like, such as vgg or alexnet
            pass
        self.model.fc.weight.data.normal_(0, 0.005)
        self.model.fc.bias.data.fill_(0.1)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self.forward(x)

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


def get_video_details(video_array, model):
    # video_array(frame_num, w, h, channel)
    crop = Rescale((224, 224))
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    print(video_array.shape)
    preprocess_array = None 
    results = None
    batch_size = 10
    batch_num = int(video_array.shape[0]/batch_size)
    for i, frame in enumerate(video_array):
        crop_image = crop(frame)
        tensor_image = to_tensor(crop_image).cuda()
        # print(tensor_image.shape)
        norm_image = norm(tensor_image)
        norm_image = norm_image.float()
        norm_image = norm_image.unsqueeze(0)
        if(preprocess_array == None):
            # print("new!")
            preprocess_array = norm_image
        else:
            # print("append!")
            preprocess_array = torch.cat([preprocess_array, norm_image], dim=0)
    print(preprocess_array.shape)

    for i in range(batch_num):
        if(results==None):
            results = model(preprocess_array[i*batch_size:(i+1)*batch_size])
        else:
            result_now = model(preprocess_array[i*batch_size:(i+1)*batch_size])
            results = torch.cat([results,result_now],axis = 0)
    
    if(results==None):
        results = model(preprocess_array[batch_num*batch_size:])
    elif(video_array.shape[0]-batch_size*batch_num >0):
        # print(preprocess_array[batch_num*batch_size:].shape)
        result_now = model(preprocess_array[batch_num*batch_size:])
        results = torch.cat([results,result_now],axis = 0)

    return results

def get_image_details(np_image,model):
    # image_array(w,h,channel)

    crop = Rescale((224, 224))
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    crop_image = crop(np_image)
    tensor_image = to_tensor(crop_image).cuda()
    # print(tensor_image.shape)
    norm_image = norm(tensor_image)
    norm_image = norm_image.float()
    norm_image = norm_image.unsqueeze(0)

    result = model(norm_image)

    return result