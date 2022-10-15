import os, sys
sys.path.append(os.getcwd())
from classes.video import Video


video_test = Video("./source/video/1_4.mp4")
print(video_test.name)
print(video_test.path)
print(video_test.sample_time_interval)
print(video_test.frame_count)
print(video_test.frames[0].frame_np.shape)