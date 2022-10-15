
import os, sys

sys.path.append(os.getcwd())

from tools.video_features.alphapose import calc_pose_of_videos_v2
from tools.video_features.roi import calc_roi_of_videos
from tools.video_features.combine_pose_roi import combine_roi_pose, calc_features
from classes.video import Video

video_test = Video("./source/test_cuts/videos/1_4.mp4")

calc_pose_of_videos_v2([video_test])
calc_roi_of_videos([video_test])
combine_roi_pose([video_test])

print(video_test.frames[0].comb_data)

calc_features([video_test])

print(video_test.frames[0].view)
