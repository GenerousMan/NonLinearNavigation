import cv2
import os

classes = ['Up/','Low/','NeckLine/','Sheet/','Waist/','Hem/','None/']

vid_path = './all_videos/'

img_path = './all_images/'

for cl in classes:
    class_path = vid_path+cl
    videos = os.listdir(class_path)
    print(cl)
    for vid in videos:
        print(vid)
        if(vid.split('.')[-1]!='mp4'):
            continue
        cap = cv2.VideoCapture(class_path+vid)
        count = 0
        while True:
            # get a frame
            ret, image = cap.read()
            if image is None:
                break
            # show a frame
            if(count % 6 ==0):
                cv2.imwrite(img_path+cl+vid.split('.')[0]+'-'+str(count/6)+'.png',image)
            count+=1
