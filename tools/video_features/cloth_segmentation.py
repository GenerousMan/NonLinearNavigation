import os
import sys
sys.path.append(os.getcwd())
import sys
sys.path.append(os.getcwd()+"/libs/")
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2
from libs.unet_segmentation.prediction.display import display_prediction,display_prediction_cv2
from libs.unet_segmentation.unet import Unet
from libs.unet_segmentation.prediction.post_processing import (
    prediction_to_classes,
    mask_from_prediction,
    remove_predictions
)


from classes.video import Video

colors = [
            [60,20,220], # 猩红
            [205,0,0], # 蓝色
            [113,179,60], # 绿色
            [0,255,255], # 纯黄
            [0,165,255], # 橙色
            [192,192,192] # 灰色
        ]

DEEP_FASHION2_CUSTOM_CLASS_MAP = {
    1: "trousers", # 裤子
    2: "skirt", # 裙子
    3: "top", # 上衣
    4: "dress", # 连衣裙
    5: "outwear", # 外套
    6: "shorts" # 短裤
}


def preprocess_image(image, image_size=512) -> torch.Tensor:
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    preprocess_pipeline = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(1,))
    ])
    return preprocess_pipeline(image)

def get_seg_tensor(frames_numpy):
    frames_tensor = [preprocess_image(frames_numpy[i]) for i in range(len(frames_numpy))]
    frames_tensor = torch.stack(frames_tensor, dim=0)
    return frames_tensor

def unet_predict(unet: Unet, img_tensor, img_numpy,
                label_map: dict = DEEP_FASHION2_CUSTOM_CLASS_MAP,
                device: str = 'cuda',
                min_area_rate: float = 0.0,
                batch_size = 2):
    img_tensor = img_tensor.to(device)

    batch_num = img_tensor.shape[0]//batch_size
    print(img_tensor.shape[0], batch_size, batch_num)
    # 批处理
    if(img_tensor.shape[0] < batch_size):
        prediction_map_all = torch.argmax(unet(img_tensor), dim=1).squeeze(0).cpu().numpy()
    else:
        prediction_map_all = torch.argmax(unet(img_tensor[:batch_size]), dim=1).squeeze(0).cpu().numpy()
        
        for i in range(1, batch_num):
            prediction_map_each = \
                    torch.argmax(unet(img_tensor[i*batch_size:(i+1)*batch_size]), dim=1).squeeze(0).cpu().numpy()
            prediction_map_all = np.vstack((prediction_map_all,prediction_map_each))
        
        if(img_tensor.shape[0]>batch_num*batch_size):

            prediction_map_each = \
                    torch.argmax(unet(img_tensor[batch_num*batch_size:]), dim=1).cpu().numpy()
            # print(prediction_map_all.shape, prediction_map_each.shape)
            prediction_map_all = np.vstack((prediction_map_all,prediction_map_each))
    
    classes_in_video = []
    mask_in_video = []
    print(prediction_map_all.shape[0],len(img_numpy))
    # Remove spurious classes
    for i in range(prediction_map_all.shape[0]):
        prediction_map = prediction_map_all[i]
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
        used_colors = [colors[predicted_classes[i].class_id-1]for i in range(len(predicted_classes))]
        img = img_numpy[i]
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        mask_all = []
        final_image = np.zeros(img.size[0]*img.size[1]*3)
        final_image.shape=[img.size[1],img.size[0],3]

        for i,mask in enumerate(masks):
            mask_now = mask.resize(img).binary_mask.astype(np.uint8)
            mask_all.append(mask_now)
            final_image = final_image+np.expand_dims(mask_now,axis=2)*[used_colors[i]]
        classes_in_video.append(predicted_classes)
        mask_in_video.append(mask_all) 
    
    return classes_in_video,mask_in_video

def calc_seg_of_videos(videos):
    torch.cuda.empty_cache()
    model = torch.load('libs/unet_segmentation/unet_iter_1300000.pt')

    for video in videos:
        frames_numpy = [video.frames[i].frame_np for i in range(len(video.frames))]
        frames_tensor = get_seg_tensor(frames_numpy)
        # print(frames_tensor.shape)
        classes, masks = unet_predict(model, frames_tensor, frames_numpy)
        # print(len(classes), len(masks),len(video.frames))

        mask_max = 0
        for i in range(len(video.frames)):
            # print("------frame:",i)
            class_frame = classes[i]
            mask_frame = masks[i]
            for j,id in enumerate(class_frame):
                id = id.class_id-1
                # print("id:",id)
                mask = mask_frame[j]
                mask_sum = np.sum(mask)
                back_sum = np.sum(1-mask)

                mask_color =  np.stack((mask,mask,mask),axis = 2)*video.frames[i].frame_np
                back_color =  (1-np.stack((mask,mask,mask),axis = 2))*video.frames[i].frame_np
                if(mask_sum == 0):
                    video.frames[i].cloth_color[id] = np.array([0,0,0])
                    mask_r = 0
                    mask_g = 0
                    mask_b = 0
                    back_r = 0
                    back_g = 0
                    back_b = 0

                # print(mask_color.shape)

                else:
                    mask_r = np.sum(mask_color[:,:,0])/mask_sum
                    mask_g = np.sum(mask_color[:,:,1])/mask_sum
                    mask_b = np.sum(mask_color[:,:,2])/mask_sum
                    back_r = np.sum(back_color[:,:,0])/back_sum
                    back_g = np.sum(back_color[:,:,1])/back_sum
                    back_b = np.sum(back_color[:,:,2])/back_sum

                    video.frames[i].cloth_color[id] = np.array([mask_r,mask_g,mask_b])
                if(mask_sum>=mask_max):
                    mask_max=mask_sum
                    video.frames[i].cloth_max_color = np.array([mask_r,mask_g,mask_b])
                    video.frames[i].back_color = np.array([back_r,back_g,back_b])
                    # print(video.frames[i].cloth_color[id])



def get_video_seg(model, video_path, debug = False):
    torch.cuda.empty_cache()
    video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    width = int(video.get(3))  # float
    height = int(video.get(4))  # float
    length = video.get(7)
    if(debug):
        if(length>400):
            return
    out_path = "output/"+video_path.split("/")[-1]
    # print(out_path)
    out = cv2.VideoWriter(out_path, fourcc, 24, (width, height), True)
    count = 0
    while (True):
        ret, f = video.read()
        if ret:
            count+=1
            # if(count%50==0):
            #     print(count, "/", length)
            predicted_classes,mask_all,final_image = display_prediction_cv2(model, f)
            # print(predicted_classes, np.stack((mask_all[0],mask_all[0],mask_all[0]),axis = 2)*f)
            # print(final_image.dtype)
            out.write((f*0.5+final_image*0.5).astype(np.uint8))
        else:
            # print("Nothing read")
            break
    out.release()

if __name__ =="__main__":

    calc_seg_of_videos([Video("./source/video/1_12.mp4")])

    # video_dir = "./source/video/"
    # files = os.listdir(video_dir)
    # for video in files:
    #     print("[ INFO ] Calculating: ", video)
    #     path_all = video_dir+video
    #     model = torch.load('libs/unet_segmentation/unet_iter_1300000.pt')
    #     # display_prediction_cv2(model, cv2.imread('test_seg_7.jpg'))
    #     get_video_seg(model, path_all)
    #     # get_video_seg(model, "./test_video.mp4")
