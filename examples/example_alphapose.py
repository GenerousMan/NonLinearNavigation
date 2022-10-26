import os, sys

sys.path.append(os.getcwd())

from tools.video_features.alphapose import calc_pose_of_videos_v2
from classes.video import Video

video_test = Video("./source/video/1_4.mp4")

calc_pose_of_videos_v2([video_test])

print(video_test.frames[0].alphapose)