import os, sys
import pickle
import cv2
from moviepy.editor import *
sys.path.append(os.getcwd())

from classes.graph import Graph
from classes.video import Video
from tools.graph_process.graph_features import *
from tools.graph_process.path_select import *

test_path = "/root/ali-2021/mm_sim/source/template_video/"
video_path = test_path+"videos/"
result_path = test_path+"results/"
pickle_path = test_path+"pickle/"
path_json_file_path = test_path+"results/path.json"


# if(os.path.isfile(pickle_path+'graph.pickle')):
#         with open(pickle_path+'graph.pickle', 'rb') as f:
#             graph_video = pickle.load(f)
#             graph_video = Graph(graph_video.video_nodes,[],True)

json_clustered = graph_clustering(result_path+ "cloth_rec_99.json",
                result_path+"clustering.json")

# best_shots_path,group_rank = get_shot_path(json_clustered,graph_video)
# # 按类间顺序组织，并顺序返回每一聚类中的顺序。group rank记录了第i个播放的聚类的group序号。

# best_path_json = {group_rank[i]:best_shots_path[i] for i in range(len(best_shots_path))}

# print(best_path_json)
# path_json_file = json.dumps(best_path_json)
# f = open(path_json_file_path, 'w')
# f.write(path_json_file)
# f.close()


# size = [720,720]

# for i,cluster in enumerate(best_shots_path):
#     L = []
#     for video_each in cluster:
#         filePath = os.path.join(video_path, video_each)
#         # 载入视频
#         video = VideoFileClip(filePath).resize(size)
#         # print(video.size)
#         # 添加到数组
#         L.append(video)
#     if(len(L)>0):
#         final_clip = concatenate_videoclips(L)
#         # 生成目标视频文件
#         final_clip.to_videofile(result_path+str(i)+".mp4", fps=24, remove_temp=False)