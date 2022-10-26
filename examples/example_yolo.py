import os, sys

sys.path.append(os.getcwd())

from tools.video_features.roi import *
from classes.video import Video

video_test = Video("./test/test_input/test_full_video.mp4")
calc_roi_of_videos([video_test])
print(video_test.frames[0].roi)
