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

def cosine_similarity(x, y, norm=True):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


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


def get_two_score(edges, clustering, base_video, random_video):

    # base_video = 103
    base_video_name = 'taobao-'+str(base_video)+".mp4"
    neb_all = [edge['target'] for edge in edges if edge['source']== base_video_name]
    print("now base:", base_video_name)
    print(len(neb_all))

    random_video_name = 'taobao-'+str(random_video)+".mp4"
    current_video = base_video
    pools = {}
    cluster_count = 0
    for edge in edges:
        if edge['source']==random_video_name and edge['target'] in neb_all:
            cluster_count+=1
    simi_cc = clustering[random_video_name]*clustering[base_video_name]*cluster_count/len(neb_all) 
    print(simi_cc)

    return simi_cc


if __name__ =="__main__":
    test_path = "/root/ali-2021/mm_sim/source/template_video/"
    video_path = test_path+"videos/"
    result_path = test_path+"results/"
    pickle_path = test_path+"pickle/"

    thres_all = [0.85]

    w_alls = [ [0.3,0.2,0.5]
            ]

    for w in w_alls:
        w_list = w + [0]
        print(w_list)
        time_start=time.time()
        graph_video = init(test_path, video_path, result_path, pickle_path,w_list)
        
        time_end=time.time()
        print('totally cost',time_end-time_start)
        first = False
        for thres in thres_all:
            graph_json = get_recommend(graph_video,thres)
            nodes = graph_json['nodes']
            edges = graph_json['links']
            degree = calc_degree(graph_json)
            # print(degree)
            clustering = calc_clustering(graph_json)
            for search_round in range(100):
                search_save_path = "./results/search/search-"+str(search_round)+"/"
                if(not os.path.exists(search_save_path)):
                    os.mkdir(search_save_path)
                base_video = random.randint(0,787)
                base_video_name = 'taobao-'+str(base_video)+".mp4"
                while(not base_video_name in degree.keys() or degree[base_video_name]<3):
                    base_video = random.randint(0,787)
                    base_video_name = 'taobao-'+str(base_video)+".mp4"
                shutil.copy(video_path+base_video_name, search_save_path+"base.mp4")
                for i in range(15):
                    # 随机15次，每次选取一个与其比较。
                    random_video = random.randint(0,787)
                    random_video_name = 'taobao-'+str(random_video)+".mp4"
                    while(not random_video_name in degree.keys() or degree[random_video_name]<3):
                        random_video = random.randint(0,787)
                        random_video_name = 'taobao-'+str(random_video)+".mp4"
                    print("random:",random_video)
                    cc_score = get_two_score(edges, clustering, base_video,random_video)
                    shutil.copy(video_path+random_video_name, search_save_path+str(cc_score)+random_video_name)
                    print(base_video, random_video, cc_score)


            
       