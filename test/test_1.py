import json 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

out_path = './results/cloth_seg_4.json'
file = open(out_path,'r',encoding='utf-8')
results = json.load(file)
msk_0 = results[1]['masks'][0]
a = np.array(msk_0)
print(a.shape)
print(a.sum())

# for msk in msk_0:
#     print(set(msk))


np.save('./results/test_4-1.npy',a)

for res in results:
    print(res['classes'])

# imgDir = './source/cloth_seg/mathi'
# fileList = os.listdir(imgDir)  #获取该文件夹下所有文件名
# fileList.sort()
# print(len(fileList))

# for i in range(2):  # len(fileList)
#     img_name = fileList[i]
#     open_img = imgDir +'/'+img_name
#     print(i,"/",len(fileList),img_name)
    
#     img = cv2.imread(open_img)
#     print("img.type:",type(img))
#     # print("img:",img)
