import pickle
from typing_extensions import final
import numpy as np
import os
import random
from classes.graph import Graph
from classes.video import Video
from tools.graph_process.graph_features import *

def print_json(json_path):
    # 作统计用

    with open(json_path,"rb") as f:
        graph_json = json.load(f)
    # 对每个类别的节点作输出
    label_now = 0
    nodes_classified_all = []
    nodes_classified_vid = []
    # 对每一列节点作统计
    ret = True
    while ret:
        nodes_temp = []
        ret = False
        for i,node in enumerate(graph_json['nodes']):
            if(node['group']==label_now):
                nodes_temp.append(node['id'])
                ret = True

        vid_names_temp = [name.split("-")[0] for name in nodes_temp]
        # print(label_now,len(nodes_temp),len(Counter(vid_names_temp).keys()))
        label_now+=1
        nodes_classified_all.append(nodes_temp)
        nodes_classified_vid.append(vid_names_temp)

def evaluate_path(path,shot_objects,relation_all):
    # 不同镜头间，景别尽可能相似，视觉关系尽可能相似，时长尽可能相似
    # 先找path里每个的index，去索引relation和object
    index_path = []
    for shot in path:
        for i,object in enumerate(shot_objects):
            if(object.name==shot):
                index_path.append(i)
                break
    score = 0
    for i in range(1,len(path)):
        # human和visual的相似度得分越高越好
        score += relation_all['human'][index_path[i]][index_path[i-1]]
        score += relation_all['visual'][index_path[i]][index_path[i-1]]

        # 来源如果相同要扣分，上下镜头时长差的越多扣的越多
        score -= relation_all['source'][index_path[i]][index_path[i-1]]
        score -= relation_all['duration'][index_path[i]][index_path[i-1]]
    return score

def generate_path(shots, shot_objects,relation_all):
    # 根据输入的shots进行重新生成。
    # 可能还需要对镜头作筛选。
    random.shuffle(shots)
    return shots

def get_cluster_path(label, name_list,graph):
    # 在cluster内决定路径, 并产出该cluster的特征
    return_shots = name_list

    if(label==1):
        best_path = name_list

    else:
        index_list = [i for i in range(len(graph.video_nodes)) if graph.video_nodes[i].name in name_list]
        shot_objects = [graph.video_nodes[i] for i in range(len(graph.video_nodes)) if graph.video_nodes[i].name in name_list]
        # print("index:", index_list)
        human_shots_relation = np.zeros((len(name_list),len(name_list)))
        visual_shots_relation = np.zeros((len(name_list),len(name_list)))
        duration_shots_relation = np.zeros((len(name_list),len(name_list)))
        source_shots_relation = np.zeros((len(name_list),len(name_list)))
        for i in range(len(index_list)):
            for j in range(i,len(index_list)):
                human_shots_relation[i][j] = graph.human_matrix[index_list[i]][index_list[j]]
                human_shots_relation[j][i] = graph.human_matrix[index_list[j]][index_list[i]]
                visual_shots_relation[i][j] = graph.visual_matrix[index_list[i]][index_list[j]]
                visual_shots_relation[j][i] = graph.visual_matrix[index_list[j]][index_list[i]]
                source_shots_relation[i][j] = (graph.video_nodes[index_list[i]].name.split("-")[-2] == graph.video_nodes[index_list[j]].name.split("-")[-2])
                source_shots_relation[j][i] = source_shots_relation[i][j]
                duration_shots_relation[i][j] = abs(graph.video_nodes[index_list[i]].frame_count / graph.video_nodes[index_list[i]].fps \
                                                - graph.video_nodes[index_list[j]].frame_count / graph.video_nodes[index_list[j]].fps)
                duration_shots_relation[j][i] = duration_shots_relation[i][j] 
                 

        relation_all = {'human':human_shots_relation,'visual':visual_shots_relation,'duration':duration_shots_relation,'source':source_shots_relation}
        # print(relation_all)
        # 初始化shot之间的关系矩阵，都是0（代表无关）。
        # 目前有如下关系：人物、视觉、时长、来源。
        # 希望不同镜头间，相连的镜头人物相似、视觉相似、时长相似、来源不同。
        # 下面将对各个关系矩阵从图中作抽取，然后得到新的关系矩阵。


        # 目前用最原始的算法，直接shuffle
        max_iter = 100
        max_score = 0
        best_path = return_shots
        for i in range(max_iter):
            return_shots = generate_path(return_shots, shot_objects,relation_all)
            # 重复运算100次，找到得分最高的序列。
            score = evaluate_path(return_shots, shot_objects,relation_all)
            # print("iter:",i,". score:",score)
            if(score>max_score):
                best_path = return_shots

    return best_path

def get_cluster_view(cluster_blanket, graph):
    # 计算平均景别
    view_all = []
    for i,node_name in enumerate(cluster_blanket):
        for node in graph.video_nodes:
            if(node.name==node_name):
                view_score = np.mean(np.array([node.frames[i].view for i in range(len(node.frames))]))
                view_all.append(view_score)
    # print(np.mean(np.array(view_all)))
    return np.mean(np.array(view_all))


    
def cluster_rank(view_features, output_shots):
    # 决定类别间关系，给不同Cluster进行重新排序
    ranked_cluster = []
    # print(view_features)
    group_rank = []

    for i in range(len(output_shots)):
        # print(i,"'th group,total:",len(output_shots[i]))
        min_view = 1e5
        for j in range(len(output_shots)):
            if(view_features[j]<min_view):
                # 找景别最小的
                min_view = view_features[j]
                min_now = j
        # print(min_now)
        # print(view_features)
        if(len(output_shots[i])>3):
            ranked_cluster.append(output_shots[min_now])
            group_rank.append(min_now+1)

        view_features[min_now] = 1e5

    return group_rank,ranked_cluster

def additional_shots(cluster_views,shots_output,graph):
    # 把未纳入展示的镜头加入cluster中
    pass

def get_shot_path(clustered_graph_json, graph):
    # 输入一个节点分类完毕的graph,json格式
    # 返回一个list,包含了若干组镜头，每一组内有其顺序

    # 记录所有得到的镜头
    shots_flat = []

    # 分组记录
    shots_output = []
    # 输出所有的镜头，两层的list
    
    cluster_blanket = []
    # 针对每一个类别的镜头作重新排布
    
    cluster_views = []
    # 每一个cluster的景别特征均值

    nodes = clustered_graph_json['nodes']
    all_class_num = len(Counter([node['id'].split("-")[-2] for node in nodes]).keys())

    links = clustered_graph_json['links']

    label_now = 1
    ret = True
    # 针对每一类都进行单独的计算，当计算类别超出总类别数时，break
    max_label = max([node['group'] for node in nodes])
    while label_now <= max_label:
        # print(label_now)
        cluster_blanket = []
        ret = False
        for i,node in enumerate(nodes):
            # print('group:',node['group'],"label now:",(label_now))
            if(node['group']==label_now):                
                cluster_blanket.append(node['id'])
                ret = True
        if(ret):
            # print("label now:",label_now)
            # 通过cluster的重新排序，依据特征、关联等信息。更新blanket内的顺序
            cluster_blanket = get_cluster_path(label_now,cluster_blanket,graph)
            view = get_cluster_view(cluster_blanket,graph)
            shots_output.append(cluster_blanket)
            shots_flat+=cluster_blanket

            cluster_views.append(view)
        label_now += 1
        # 进行下一类计算
    print("total num of groups:",len(shots_output))
    additional_shots(cluster_views,shots_output,graph)

    group_rank, shots_output = cluster_rank(cluster_views, shots_output)
    print("final num of groups:",len(shots_output))
    final_class_num = len(Counter([shot.split("-")[-2] for shot in shots_flat]).keys())
    print("initial video num:",all_class_num,"final video num:",final_class_num)
    return shots_output,group_rank






