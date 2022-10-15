import json
from collections import Counter
import random

def calc_degree(graph_json):

    nodes = graph_json['nodes']
    edges = graph_json['links']
    degree_list = {node['id']:0 for node in nodes}
    # print(degree_list)
    i=0
    for edge in edges:
        # print(i, len(edges))
        degree_list[edge['source']] += 1
        i+=1

    return degree_list

def get_cluster(graph_json, search_round, search_step, init_id = None):
    nodes = graph_json['nodes']
    edges = graph_json['links']

    degree = calc_degree(graph_json)
    # print(degree)

    clustering = calc_clustering(graph_json)
    # print(clustering)
    base_video = init_id
    if(init_id==None):
        base_video = random.randint(0,787)
    # base_video = 103
    base_video_name = 'taobao-'+str(base_video)+".mp4"
    neb_all = [edge['target'] for edge in edges if edge['source']== base_video_name]
    print(len(neb_all))

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

    return base_video_name, pools_list


def calc_centrality(graph_json):

    pass

def calc_clustering(graph_json):
    nodes = graph_json['nodes']
    edges = graph_json['links']

    clustering_all = {}

    for i,node in enumerate(nodes):
        target_all = [edge['target'] for edge in edges if edge['source']== node['id'] ]
        # print(target_all)
        count=0
        for edge in edges:
            if edge['source'] in target_all and edge['target'] in target_all:
                count+=1
        # 双向图，除以2
        count /= 2
        total_target_num = len(target_all)
        if(total_target_num>1):
            clustering = 2 * count / ((total_target_num-1)*total_target_num)
        else:
            clustering = 0
        clustering_all.update({node["id"]:clustering})

    return clustering_all
def graph_refine(graph_json):
    refine_label_now = 1

    max_label = max([node['group'] for node in graph_json['nodes']])
    for graph_label_now in range(1,max_label+1):
        ret = False
        for i,node in enumerate(graph_json['nodes']):
            if(graph_json['nodes'][i]['group']==graph_label_now):
                # 当前数值存在
                ret = True
                break
        if(ret):

            # 存在当前数值，说明可以把当前数值用refine的标签代替
            if(refine_label_now == graph_label_now):
            # 不用代替
                continue
            else:
                # print("label:",graph_label_now,"->",refine_label_now)
                for i,node in enumerate(graph_json['nodes']):
                    if(graph_json['nodes'][i]['group']==graph_label_now):
                        # 当前数值存在,则替换为refine的标签
                        graph_json['nodes'][i]['group'] = refine_label_now
                refine_label_now += 1
        # 不存在则直接跳过当前值
    return graph_json

def graph_clustering(graph_json_path, graph_json_path_new, center_degree = 10, center_clust = 0.5, cluster_degree = 4, cluster_clust = 0.6):
    with open(graph_json_path,"rb") as f:
        graph_json = json.load(f)

    # print(graph_json)

    degree = calc_degree(graph_json)
    print(degree)

    clustering = calc_clustering(graph_json)
    print(clustering)

    nodes = graph_json['nodes']
    edges = graph_json['links']
    clustering_label_now = 2
    # 自适应确定类别，每确定一类就多一类。
    for i,node in enumerate(nodes+nodes):
        # 阈值可以自己设置。
        # 计算两遍，可以避免有的单节点计算出错。
        if(i>=len(nodes)):
            i-=len(nodes)
        if(degree[node['id']]>center_degree and clustering[node['id']]<center_clust):
            # 核心节点
            graph_json['nodes'][i]['group'] = 1

        if(clustering[node['id']]>cluster_clust and degree[node['id']]>cluster_degree):
            # 聚类节点，考虑其直接相连节点的类别，自动确定类别。
            target_all = [edge['target'] for edge in edges if edge['source'] == node['id']]
            target_label_all = [node_now['group'] for node_now in nodes if(node_now['id'] in target_all)]
            target_label_count = Counter(target_label_all)
            max_count_label = target_label_count.most_common(1)[0][0]
            max_count = target_label_count.most_common(1)[0][1]
            # 统计链接节点的类别
            if(max_count_label!=0 and max_count >= 0.6*len(target_label_all)):
                # 链接的节点，有一半以上已经确定是某类了，则全部设置为该类。
                now_label = target_label_count.most_common(1)[0][0]
            else:
                now_label = clustering_label_now
                clustering_label_now+=1
            # print("now we choose:",now_label)
            # print(node['id'],now_label)
            # print(target_label_all)
            graph_json['nodes'][i]['group'] = now_label
            for j,target in enumerate(nodes):
                # print(target)
                if(target['id'] in target_all and  clustering[target['id']]>0.6 and graph_json['nodes'][j]['group'] != 1 ):
                    graph_json['nodes'][j]['group'] = now_label

    # 把多余标签值删去
    graph_json = graph_refine(graph_json)
    new_json_file = json.dumps(graph_json)
    f = open(graph_json_path_new, 'w')
    f.write(new_json_file)
    f.close()
    return graph_json
