import os, sys
import pickle
import cv2
from moviepy.editor import *
sys.path.append(os.getcwd())

from classes.graph import Graph
from classes.video import Video
from tools.graph_process.graph_features import *
from tools.graph_process.path_select import *
import random

def get_cluster(graph_json):
    nodes = graph_json['nodes']
    edges = graph_json['links']

    degree = calc_degree(graph_json)
    # print(degree)

    clustering = calc_clustering(graph_json)
    # print(clustering)

    # base_video = random.randint(0,787)
    base_video = 103
    base_video_name = 'taobao-'+str(base_video)+".mp4"
    neb_all = [edge['target'] for edge in edges if edge['source']== base_video_name]
    print(neb_all)

    while(len(neb_all)<30):
        base_video = random.randint(0,787)
        base_video_name = 'taobao-'+str(base_video)+".mp4"
        neb_all = [edge['target'] for edge in edges if edge['source']== base_video_name]
    print("now base:", base_video_name)
    print(len(neb_all))

    current_video = base_video
    pools = {}
    for i in range(search_round):
        current_video_name = 'taobao-'+str(current_video)+".mp4"
        print("search round:",i,"/",search_round)
        while(not current_video_name in degree.keys() or degree[current_video_name]<3):
            current_video = random.randint(0,787)
            current_video_name = 'taobao-'+str(current_video)+".mp4"
        print("now video:", current_video_name,'degree:',degree[current_video_name])
        for j in range(search_step):
            target_all = [edge['target'] for edge in edges if edge['source']== current_video_name ]
            target_random_id = random.randint(0,len(target_all))
            target_random_name = 'taobao-'+str(target_random_id)+".mp4"
            while(not target_random_name in clustering.keys()):
                target_all = [edge['target'] for edge in edges if edge['source']== current_video_name ]
                target_random_id = random.randint(0,len(target_all))
                target_random_name = 'taobao-'+str(target_random_id)+".mp4"

            cluster_count = 0
            for edge in edges:
                if edge['source']==target_random_name and edge['target'] in neb_all:
                    cluster_count+=1
            simi_cc = clustering[target_random_name]*clustering[base_video_name]*cluster_count/len(neb_all) 
            print(simi_cc)
            pools.update({target_random_name:simi_cc})
            pools_list = sorted(pools.items(), key=lambda item:item[1], reverse=True)

            current_video_name = pools_list[0][0]
            # print(len(target_all))
            # search_video_name = 'taobao-'+str(search_id)+".mp4"

    print(base_video_name)
    print(pools_list)

test_path = "/root/ali-2021/mm_sim/source/template_video/"
video_path = test_path+"videos/"
result_path = test_path+"results/"
pickle_path = test_path+"pickle/"
path_json_file_path = test_path+"results/path.json"

search_round = 5
search_step = 10

graph_json_path = result_path+ "cloth_rec.json"
with open(graph_json_path,"rb") as f:
        graph_json = json.load(f)

get_cluster(graph_json)