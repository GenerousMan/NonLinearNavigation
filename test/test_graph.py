import os, sys

from torch.nn.functional import threshold

sys.path.append(os.getcwd())

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

if __name__ =="__main__":
    # 对指定文件夹的所有视频进行统一计算
    test_path = "/root/ali-2021/mm_sim/source/template_video/"
    video_path = test_path+"videos/"
    result_path = test_path+"results/"
    pickle_path = test_path+"pickle/"

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
            graph_video = Graph(graph_video.video_nodes,[],True)
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


    print(len(graph_video.video_nodes))
    graph_video.video_nodes[0].frames[0].print_self()

    graph_test = graph_video#Graph(graph_video.video_nodes, graph_video.text_nodes, True)

    # 绘制热力图
    sns.heatmap(graph_test.human_matrix, cmap='Reds')
    plt.savefig(result_path+"human")

    plt.clf()

    sns.heatmap(graph_test.cloth_matrix, cmap='Reds')
    plt.savefig(result_path+"cloth")

    plt.clf()

    sns.heatmap(graph_test.visual_matrix, cmap='Reds')
    plt.savefig(result_path+"visual")

    plt.clf()
    
    # # 绘制直方图均衡化后的热力图
    # graph_test.human_matrix = exposure.equalize_hist(graph_test.human_matrix)
    # graph_test.cloth_matrix = exposure.equalize_hist(graph_test.cloth_matrix)
    # graph_test.visual_matrix = exposure.equalize_hist(graph_test.visual_matrix)
    

    # sns.heatmap(graph_test.human_matrix, cmap='Reds')
    # plt.savefig(result_path+"human_equalize_hist")

    # plt.clf()

    # sns.heatmap(graph_test.cloth_matrix, cmap='Reds')
    # plt.savefig(result_path+"cloth_equalize_hist")

    # plt.clf()

    # sns.heatmap(graph_test.visual_matrix, cmap='Reds')
    # plt.savefig(result_path+"visual_equalize_hist")

    # mds_visual = MDS(graph_test.visual_matrix)
    # mds_human = MDS(graph_test.human_matrix)

    # plt.clf()
    # plt.scatter(mds_visual[:,0],mds_visual[:,1])
    # plt.savefig(result_path+"visual_mds")

    # plt.clf()
    # plt.scatter(mds_human[:,0],mds_human[:,1],alpha=0.5)
    # plt.savefig(result_path+"human_mds")




    # 设定筛选的有效边值
    human_threshold = 0.95
    cloth_threshold = 0.99
    visual_threshold = 0.99

    f=open(result_path+"human_simi.txt","w")
    f2=open(result_path+"cloth_simi.txt","w")

    for i in range(len(graph_test.video_nodes)):
        print(i,graph_test.video_nodes[i].name)
        graph_test.video_nodes[i].avg_view = int(0.5 + np.mean(np.array(
            [graph_test.video_nodes[i].frames[j].view for j in range(len(graph_test.video_nodes[i].frames))])))
        graph_test.video_nodes[i].avg_direction = int(0.5 + np.mean(np.array(
            [graph_test.video_nodes[i].frames[j].direction for j in range(len(graph_test.video_nodes[i].frames))])))
        graph_test.video_nodes[i].avg_pose = int(0.5 + np.mean(np.array(
            [graph_test.video_nodes[i].frames[j].pose for j in range(len(graph_test.video_nodes[i].frames))])))
        graph_test.video_nodes[i].avg_motion = int(0.5 + np.mean(np.array(
            [graph_test.video_nodes[i].frames[j].motion for j in range(len(graph_test.video_nodes[i].frames))])))

    # 计算关联矩阵，绘制json点线图
    group_type = {
        'Tee':0,
        'coat':1,
        'dress':2,
        'jacket':3,
        'jeans':4,
        'shorts':5,
        'sweater':6,
        'blazer':7,
        'shirt':8,
        'fur':9,
        'hoodies':10,

    }

    human_json_data = {'nodes':[], 'links':[]}
    cloth_json_data = {'nodes':[], 'links':[]}
    visual_json_data = {'nodes':[], 'links':[]}
    human_cloth_json_data = {'nodes':[], 'links':[]}

    edge_weight = [0,1,3,8]
    for i in range(len(graph_test.video_nodes)):
        # human_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':group_type[graph_test.video_nodes[i].name.split("_")[0]]})
        # visual_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':group_type[graph_test.video_nodes[i].name.split("_")[0]]})
        # cloth_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':group_type[graph_test.video_nodes[i].name.split("_")[0]]})
        # human_cloth_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':group_type[graph_test.video_nodes[i].name.split("_")[0]]})

        human_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':0})
        visual_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':0})
        cloth_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':0})
        human_cloth_json_data['nodes'].append({'id':graph_test.video_nodes[i].name, 'group':0})
        for j in range(i+1, len(graph_test.video_nodes)):
            edge_type = 0
            # 两两不相似
            if(graph_test.human_matrix[i][j]>=human_threshold):
                # print("\nhuman:", graph_test.video_nodes[i].name,graph_test.video_nodes[j].name)
                f.write("\n\nhuman:"+ graph_test.video_nodes[i].name+" "+graph_test.video_nodes[j].name +" "+str(graph_test.human_matrix[i][j]))
                f.write("\nhuman features:\nview: "+ str(graph_test.video_nodes[i].avg_view) + " dire:"+ str(graph_test.video_nodes[i].avg_direction) + " pose:"+str(graph_test.video_nodes[i].avg_pose) + " motion:"+str(graph_test.video_nodes[i].avg_motion))
                f.write("\nview:"+str(graph_test.video_nodes[j].avg_view) + " dire:"+ str(graph_test.video_nodes[j].avg_direction) + " pose:"+str(graph_test.video_nodes[j].avg_pose) + " motion"+str(graph_test.video_nodes[j].avg_motion))
                human_json_data['links'].append({
                    'source':graph_test.video_nodes[i].name,
                    'target':graph_test.video_nodes[j].name,
                    'value': int(math.pow(1.1,graph_test.human_matrix[i][j]))})
                
                human_json_data['links'].append({
                    'source':graph_test.video_nodes[j].name,
                    'target':graph_test.video_nodes[i].name,
                    'value': int(math.pow(1.1,graph_test.human_matrix[i][j]))})
                edge_type += 1
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
                
            if(graph_test.visual_matrix[i][j]>=visual_threshold):
                # print("\ncloth:", graph_test.video_nodes[i].name,graph_test.video_nodes[j].name)
                visual_json_data['links'].append({
                    'source':graph_test.video_nodes[i].name,
                    'target':graph_test.video_nodes[j].name,
                    'value': math.pow(1.1,graph_test.visual_matrix[i][j])})
                visual_json_data['links'].append({
                    'source':graph_test.video_nodes[j].name,
                    'target':graph_test.video_nodes[i].name,
                    'value': math.pow(1.1,graph_test.visual_matrix[i][j])})
                edge_type += 2
                # f2.write("\ncloth features:\nfirst attribute:"+ str(graph_test.video_nodes[i].avg_attr) +"\nsecond attribute:"+ str(graph_test.video_nodes[j].avg_attr))
                # f2.write("\ncloth features:\nfirst cate:"+ str(graph_test.video_nodes[i].avg_cate) +"\nsecond cate:"+ str(graph_test.video_nodes[j].avg_cate))
                # f2.write("\ncloth features:\nfirst color:"+ str(graph_test.video_nodes[i].avg_color) +"\nsecond color:"+ str(graph_test.video_nodes[j].avg_color))
            if(edge_type != 0):
                human_cloth_json_data['links'].append({
                    'source':graph_test.video_nodes[i].name,
                    'target':graph_test.video_nodes[j].name,
                    'value': edge_weight[edge_type]})
                human_cloth_json_data['links'].append({
                    'source':graph_test.video_nodes[j].name,
                    'target':graph_test.video_nodes[i].name,
                    'value': edge_weight[edge_type]})
    f.close()
    f2.close()

    # print(human_json_data)

    human_json_file = json.dumps(human_json_data)
    f = open(result_path+'human_json_'+str(int(human_threshold*100))+'.json', 'w')
    f.write(human_json_file)
    f.close()

    cloth_json_file = json.dumps(cloth_json_data)
    f = open(result_path+'cloth_json_'+str(int(cloth_threshold*100))+'.json', 'w')
    f.write(cloth_json_file)
    f.close()


    visual_json_file = json.dumps(visual_json_data)
    f = open(result_path+'visual_json_'+str(int(visual_threshold*100))+'.json', 'w')
    f.write(visual_json_file)
    f.close()

    human_cloth_json_file = json.dumps(human_cloth_json_data)
    # f = open(result_path+'human_visual_json_'+str(int(human_threshold*100))+'_'+str(int(visual_threshold*100))+'.json', 'w')
    f = open(result_path+'exp_json.json', 'w')
    f.write(human_cloth_json_file)
    f.close()