import os, sys

sys.path.append(os.getcwd())

from tools.video_features.alphapose import calc_pose_of_videos_v2
from tools.video_features.cloth_features import calc_cloth_of_videos
from classes.video import Video

video_test = Video("/root/ali-2021/mm_sim/source/products/videos/blazer_0.mp4")

calc_cloth_of_videos([video_test])

print(video_test.frames[0].cloth_attr.shape)