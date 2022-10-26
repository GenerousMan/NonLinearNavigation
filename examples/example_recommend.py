import os, sys
import random
from torch.nn.functional import threshold

sys.path.append(os.getcwd())
import time
import pickle

from classes.graph import Graph
from classes.video import Video
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import data,exposure
import math
import numpy as np
import json
import shutil
from tools.data_process.MDS import MDS
from tools.graph_process.graph_features import *

def cosine_similarity(x, y, norm=True):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内



def calc_vv_cloth(v_node_a, v_node_b):
    cloth_attr_w = 0.2
    cloth_cate_w = 0.5
    cloth_color_w = 0.3

    attr_a_array = np.array([v_node_a.frames[i].cloth_attr for i in range(len(v_node_a.frames))])
    attr_b_array = np.array([v_node_b.frames[i].cloth_attr for i in range(len(v_node_b.frames))])

    avg_attr_a = np.mean(attr_a_array, axis = 0).tolist()
    avg_attr_b = np.mean(attr_b_array, axis = 0).tolist()

    cate_a_array = np.array([v_node_a.frames[i].cloth_cate for i in range(len(v_node_a.frames))])
    cate_b_array = np.array([v_node_b.frames[i].cloth_cate for i in range(len(v_node_b.frames))])

    avg_cate_a = np.mean(cate_a_array, axis = 0).tolist()
    avg_cate_b = np.mean(cate_b_array, axis = 0).tolist()

    color_a_array = np.array([(v_node_a.frames[i].cloth_color).reshape(18) for i in range(len(v_node_a.frames))])
    color_b_array = np.array([(v_node_b.frames[i].cloth_color).reshape(18) for i in range(len(v_node_b.frames))])

    avg_color_a = np.mean(color_a_array, axis = 0).tolist()
    avg_color_b = np.mean(color_b_array, axis = 0).tolist()

    # print(avg_color_a)

    dis_attr = cosine_similarity(avg_attr_a, avg_attr_b)
    dis_cate = cosine_similarity(avg_cate_a, avg_cate_b)
    dis_color = cosine_similarity(avg_color_a, avg_color_b)

    # print("cloth similarity:", dis_attr, dis_cate, dis_color)

    return dis_attr,dis_cate,dis_color

def get_recommend(graph_video, thres):
    graph_test = graph_video#Graph(graph_video.video_nodes, graph_video.text_nodes, True)
    graph_test.cloth_matrix = exposure.equalize_hist(graph_test.cloth_matrix)

    out_thres = thres

    keep_index = graph_test.cloth_matrix.shape[0]*graph_test.cloth_matrix.shape[1]*out_thres
    whole_cloth_values = graph_test.cloth_matrix.flatten()
    # print(whole_cloth_values)
    sort_np = np.sort(whole_cloth_values)
    i = graph_test.cloth_matrix.shape[0] * graph_test.cloth_matrix.shape[1]
    while(i>keep_index):
        i-=1
        # print(sort_np[i])

    cloth_threshold = sort_np[i]
    # print(cloth_threshold)

    cloth_json_data = {'nodes':[], 'links':[]}
    f2=open(result_path+"cloth_simi.txt","w")
    for i in range(len(graph_test.video_nodes)):
        # human_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':0})
        # visual_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':0})
        cloth_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':0})
        # human_cloth_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':0})
        for j in range(i+1, len(graph_test.video_nodes)):
            if(graph_test.cloth_matrix[i][j]>=cloth_threshold):
                attr_dis, cate_dis, color_dis = calc_vv_cloth(graph_test.video_nodes[i],graph_test.video_nodes[j])
                # print("\ncloth:", graph_test.video_nodes[i].name,graph_test.video_nodes[j].name)
                f2.write("\n\ncloth:"+ graph_test.video_nodes[i].name+" "+graph_test.video_nodes[j].name+" "+str(graph_test.cloth_matrix[i][j]))
                f2.write("\n attr dis:"+str(attr_dis)+" cate dis:"+ str(cate_dis)+" color dis:" +str(color_dis))
                cloth_json_data['links'].append({
                    'source':graph_test.video_nodes[i].name,
                    'target':graph_test.video_nodes[j].name,
                    'value': math.pow(1.1,graph_test.cloth_matrix[i][j])})
                cloth_json_data['links'].append({
                    'source':graph_test.video_nodes[j].name,
                    'target':graph_test.video_nodes[i].name,
                    'value': math.pow(1.1,graph_test.cloth_matrix[i][j])})
                # f2.write("\ncloth features:\nfirst attribute:"+ str(graph_test.video_nodes[i].avg_attr) +"\nsecond attribute:"+ str(graph_test.video_nodes[j].avg_attr))
                # f2.write("\ncloth features:\nfirst cate:"+ str(graph_test.video_nodes[i].avg_cate) +"\nsecond cate:"+ str(graph_test.video_nodes[j].avg_cate))
                # f2.write("\ncloth features:\nfirst color:"+ str(graph_test.video_nodes[i].avg_color) +"\nsecond color:"+ str(graph_test.video_nodes[j].avg_color))
    f2.close()

    # print(human_json_data)

    cloth_json_file = json.dumps(cloth_json_data)
    f = open(result_path+'cloth_rec.json', 'w')
    f.write(cloth_json_file)
    f.close()

    return cloth_json_data

def init(test_path, video_path, result_path, pickle_path,w_list):
    if(not os.path.exists(video_path)):
        os.mkdir(video_path)
        outside_videos = os.listdir(test_path)
        for video in outside_videos:
            print(video)
            if(video.split(".")[-1]=="mp4"):
                shutil.move(test_path+video,video_path+video)

    if(not os.path.exists(result_path)):
        os.mkdir(result_path)

    if(not os.path.exists(pickle_path)):
        os.mkdir(pickle_path)

    video_list = os.listdir(video_path)
    video_all = []

    if(os.path.isfile(pickle_path+'graph.pickle')):
        with open(pickle_path+'graph.pickle', 'rb') as f:
            graph_video = pickle.load(f)
            graph_video = Graph(graph_video.video_nodes,[],True, w_list)
    else:

        if(os.path.isfile(pickle_path+'video_nodes.pickle')):
            with open(pickle_path+'video_nodes.pickle', 'rb') as f:
                video_all = pickle.load(f)
        else:
            for video_now in video_list:
                if(video_now.split(".")[-1]!="mp4"):
                    continue
                video_all.append(Video(video_path+video_now)) 

            with open(pickle_path+'video_nodes.pickle', 'wb') as f:
                pickle.dump(video_all, f)
                
        graph_video = Graph(video_all,[])

        graph_video.video_nodes[0].frames[0].print_self()
        with open(pickle_path+'graph.pickle', 'wb') as f:
            pickle.dump(graph_video,f)
    return graph_video

if __name__ =="__main__":
    # 对指定文件夹的所有视频进行统一计算
    test_path = "/root/ali-2021/mm_sim/source/template_video/"
    video_path = test_path+"videos/"
    result_path = test_path+"results/"
    pickle_path = test_path+"pickle/"
    rec_path = "./results/recmmend/"

    thres_all = [0.95,0.9,0.85,0.8]

    w_alls = [
                [0.8,0,0.2],[0,0.8,0.2],[0.2,0,0.8],
                [0.6,0,0.4],[0.2,0.6,0.2],[0.4,0,0.6],
                [0.6,0.2,0.2],[0.3,0.4,0.3],[0.2,0.2,0.6]
            ]
    rounds = [3, 5, 20]
    steps = [5, 10, 20]
    first = True

    for w in w_alls:
        w_list = w + [0]
        print(w_list)
        time_start=time.time()
        graph_video = init(test_path, video_path, result_path, pickle_path,w_list)
        time_end=time.time()
        print('totally cost',time_end-time_start)
        first = False
        for thres in thres_all:
            cloth_json_data = get_recommend(graph_video,thres)
            for round in rounds:
                for step in steps:
                    print("now setting:", w, thres, round, step)
                    for i in range(3):
                        # 每组参数重复10次
                        name, recs = get_cluster(cloth_json_data, round, step)
                        f = open(rec_path+"rec.txt",'a+')
                        f.write(str(w) + " " + str(thres) + " " + str(round) + " " + str(step)+" ")
                        print(name,recs[:3])
                        f.write(name+" "+str(recs[:3]))
                        f.write("\n")
                        f.close()
       

            