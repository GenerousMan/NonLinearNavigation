from PIL import Image
import cv2
import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import libs.unet_segmentation as unet_segmentation
from libs.unet_segmentation.unet import Unet
from libs.unet_segmentation.prediction.post_processing import (
    prediction_to_classes,
    mask_from_prediction,
    remove_predictions
)
colors = [
            # [60,20,220], # 猩红
            # [205,0,0], # 蓝色
            # [113,179,60], # 绿色
            # [0,255,255], # 纯黄
            # [0,165,255], # 橙色
            # [192,192,192] # 灰色
            [255,255,255], # 猩红
            [255,255,255], # 蓝色
            [255,255,255], # 绿色
            [255,255,255], # 纯黄
            [255,255,255], # 橙色
            [255,255,255] # 灰色
        ]

DEEP_FASHION2_CUSTOM_CLASS_MAP = {
    1: "trousers", # 裤子
    2: "skirt", # 裙子
    3: "top", # 上衣
    4: "dress", # 连衣裙
    5: "outwear", # 外套
    6: "shorts" # 短裤
}


def display_prediction(
        unet: Unet,
        image_path: str,
        image_resize: int = 512,
        label_map: dict = DEEP_FASHION2_CUSTOM_CLASS_MAP,
        device: str = 'cuda',
        min_area_rate: float = 0.05) -> None:

    # Load image tensor
    img = Image.open(image_path)
    img_tensor = _preprocess_image(img, image_size=image_resize).to(device)

    # Predict classes from model
    prediction_map = \
        torch.argmax(unet(img_tensor), dim=1).squeeze(0).cpu().numpy()

    # Remove spurious classes
    classes = prediction_to_classes(prediction_map, label_map)
    predicted_classes = list(
        filter(lambda x: x.area_ratio >= min_area_rate, classes))
    spurious_classes = list(
        filter(lambda x: x.area_ratio < min_area_rate, classes))
    clean_prediction_map = remove_predictions(spurious_classes, prediction_map)

    # Get masks for each of the predictions
    masks = [
        mask_from_prediction(predicted_class, clean_prediction_map)
        for predicted_class in predicted_classes
    ]

    mask_all = np.zeros(img.size[0]*img.size[1]*3)
    plt.figure(figsize=(img.size[0]/100,img.size[1]/100))
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    # Display predictions on top of original image
    # plt.imshow(np.array(img))
    for mask in masks:
        plt.imshow(mask.resize(img).binary_mask, cmap='jet', alpha=0.7)
    plt.savefig("temp/temp_output.jpg")
    
    plt.cla()
    plt.close("all")

def display_prediction_cv2(
        unet: Unet,
        img,
        image_resize: int = 512,
        label_map: dict = DEEP_FASHION2_CUSTOM_CLASS_MAP,
        device: str = 'cuda',
        min_area_rate: float = 0.005) -> None:  #0.05
    # 输入：
    # unet: 网络结构
    # img: 图片的ndarray
    # image_resize: 图像大小
    # label_map: 输入的fashion数据集的类别
    # device: 是否用cuda
    # min_area_rate: 多大的面积范围被判定为存在该品类

    # 输出：
    # predicted_classes: 预测的衣服种类，格式为：[PredictedClass(area_ratio=0.07886886596679688, class_name='skirt', class_id=2), 
    #                           PredictedClass(area_ratio=0.0661773681640625, class_name='top', class_id=3)]
    # mask_all: 每种服装种类的mask区域，大小即为图像的 长*宽，存在即为1，不存在即为0
    # final_image: 最终的可视化结果，用了各种颜色进行可视化。
    
    # Load image tensor
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    # print(type(img))
    img_tensor = _preprocess_image(img, image_size=image_resize).to(device)

    # Predict classes from model
    prediction_map = \
        torch.argmax(unet(img_tensor), dim=1).squeeze(0).cpu().numpy()

    # Remove spurious classes
    classes = prediction_to_classes(prediction_map, label_map)
    predicted_classes = list(
        filter(lambda x: x.area_ratio >= min_area_rate, classes))
    print("predicted_classes.len:",len(predicted_classes))
    spurious_classes = list(
        filter(lambda x: x.area_ratio < min_area_rate, classes))
    clean_prediction_map = remove_predictions(spurious_classes, prediction_map)

    # Get masks for each of the predictions
    masks = [
        mask_from_prediction(predicted_class, clean_prediction_map)
        for predicted_class in predicted_classes
    ]
    

    used_colors = [colors[predicted_classes[i].class_id-1]for i in range(len(predicted_classes))]
    
    mask_all = []
    final_image = np.zeros(img.size[0]*img.size[1]*3)
    final_image.shape=[img.size[1],img.size[0],3]

    for i,mask in enumerate(masks):
        mask_now = mask.resize(img).binary_mask.astype(np.uint8)
        mask_all.append(mask_now)
        final_image = final_image+np.expand_dims(mask_now,axis=2)*[used_colors[i]]
    
    # cv2.imwrite("test_output_mask.jpg",final_image)
    return predicted_classes,mask_all,final_image.astype(np.uint8)

def _preprocess_image(image: Image, image_size: int) -> torch.Tensor:
    preprocess_pipeline = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(1,))
    ])
    return preprocess_pipeline(image).unsqueeze(0)
