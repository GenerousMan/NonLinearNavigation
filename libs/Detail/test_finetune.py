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


data_folder = '4_images'
batch_size = 32
n_class = 4

def load_data(root_path, batch_size):
    transform_dict = {
        'src': transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'tar': transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}
    data = datasets.ImageFolder(root=root_path, transform=transform_dict['src'])
    
    target = data.targets
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False,sampler = sampler)
    return data_loader

dataloader = load_data(data_folder, batch_size)
print('Source data number:', len(dataloader.dataset))

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


model = TransferModel().cuda()
model.load_state_dict(torch.load('/root/ali-2021/diy-classifier/4_model.pkl'))

model.eval()

RAND_TENSOR = torch.randn(1, 3, 224, 224).cuda()
output = model(RAND_TENSOR)
print(output)

test_image = '/root/ali-2021/diy-classifier/test_images/test_neckline2.png'
np_image = cv2.imread(test_image)[:,:,:3]
print(np_image.shape)

crop = Rescale((224, 224))
to_tensor = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

crop_image = crop(np_image)
tensor_image = to_tensor(crop_image).cuda()
print(tensor_image.shape)
norm_image = norm(tensor_image)
norm_image = norm_image.float()
norm_image = norm_image.unsqueeze(0)

result = model(norm_image)
print(result)