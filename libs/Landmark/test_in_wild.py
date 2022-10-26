import os
from arg import argument_parser
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models import Network
from utils import cal_loss, Evaluator
import utils
from skimage import io, transform
from torchvision import transforms
import cv2
from skimage import io, transform

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

def write_image(path, image_now, xy):
    # image = image_now.copy()
    # image_now = cv2.cvtColor(image_now,cv2.COLOR_RGB2BGR)
    print(image_now.shape)
    image_now = cv2.cvtColor(image_now.astype(np.float32),cv2.COLOR_RGB2BGR)
    for j in range(xy[0].shape[0]):
        print("len:",len(xy[0][j]))
        print(image_now.shape)
        print((int(image_now.shape[1]*xy[0][j][0]),int(image_now.shape[0]*xy[0][j][1])))
        cv2.circle(image_now,(int(image_now.shape[1]*xy[0][j][0]),int(image_now.shape[0]*xy[0][j][1])),10,(0,0,255),-1)
        print("draw...",j)

    image_now = image_now*255
    cv2.imwrite(path,image_now)

if __name__ == "__main__":
    net = torch.nn.DataParallel(Network(dataset=['fld'], flag=True)).cuda()
    weights = torch.load('/root/ali-2021/GLE_FLD/models/model_07.pkl')
    # weights = utils.load_weight(net, weights)
    net.load_state_dict(weights)
    image_name = "test_half.png"
    image = io.imread("/root/ali-2021/GLE_FLD/wild_image/"+image_name)[:,:,:3]/255.
    image_ori = image.copy()

    print(image)
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    to_tensor = transforms.ToTensor()
    rescale224square = Rescale((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    image = rescale224square(image)
    image = to_tensor(image)
    
    image = normalize(image)
    image = image.float()
    print(image.shape)
    image = image.unsqueeze(0)

    print(image.shape)
    image = {"image":image}

    output = net(image)
    lm_pos_map = output['lm_pos_map']
    batch_size, _, pred_h, pred_w = lm_pos_map.size()
    lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)

    lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped, dim=2).cpu().numpy(), (pred_h, pred_w))
    lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)
    print(lm_pos_output)

    write_image("./test_in_wild_results/"+image_name,image_ori,lm_pos_output)