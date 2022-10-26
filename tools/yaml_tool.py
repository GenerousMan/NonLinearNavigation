import os
import json
import yaml


detail_seq = ['neck','sheet','waist','hem']

def create_xy(x,y):
    return str(x)+' '+str(y)

def create_goto_action(event, node, to_node):
    return {'event': event, 'action': [node, 'goTo', to_node]}

def create_trans_action(event, node_name, size, position, duration):
    return  {'event': event, 'action': [node_name, 'transform', size, position, duration]}

def create_full_node(full_name):
    full_node = {'type': 'video', 'id': 'scene-1', 'url': full_name, 'overlays':[], 'actions':[]}
    return full_node

def create_show_action(event, node, duration='0ms'):
    return {'event': event, 'action': [node, 'show', duration]}

def create_overlay(detail_type, url, keypoint_list, size, time_duration):
    # 用这个函数创造overlay
    overlay_node = {'type': 'video', 'loop':True, 'id': detail_type, 'url': url, 'size': '20% 20%', 'position': '75% 5%', 'visible': False, 'styles': {'boxShadow': '0 4px 32px rgba(0, 0, 0, 0.25)', 'borderRadius': '18px'},
                    'actions': []}
    # 定义了这个Overlay的类别、url，后面就要定义这个overlay的轨迹
    show_action = create_show_action('0ms', detail_type, '250ms')
    overlay_node['actions'].append(show_action)
    for i,keypoint in enumerate(keypoint_list):
        keypoint_xy = create_xy(keypoint[0], keypoint[1])
        trans = create_trans_action(str(i*time_duration)+'ms', detail_type, size, keypoint_xy, time_duration)
        overlay_node['actions'].append(trans)

    return overlay_node

def Aniv_yaml(full_shot_name, detail_name_list, keypoints_time_list, duration):
    # 将全景作为底，将特写视频作为锚点写在关键点坐标上
    # 写20个的time action，把节点绑定在上面。
    flow = {'type': 'flow', 'id': 'root-flow', 'loop': True, 'nodes':[]}
    full_node = create_full_node(full_shot_name)
    # 创建空白的full node
    size = '100 100'
    # 根据每个类别，如果存在则创建该类别的detail浮窗node，并加至full node
    for i,name in enumerate(detail_name_list):
        if(detail_name_list[i]!=None):
            overlay = create_overlay(detail_seq[i], detail_name_list[i], keypoints_time_list[i], size, duration)
            print(overlay)
            full_node['overlays'].append(overlay)


    flow['nodes'].append(full_node)

    return flow

if __name__ == "__main__":
    full_shot_name = './video1.mp4'
    neck_shots = './video2.mp4'
    waist_shots = './video2.mp4'
    hem_shots = None
    sheet_shots = None
    print(yaml.load(open('template.yaml')))
    detail_shot_list = [neck_shots, sheet_shots, waist_shots, hem_shots]
    key_points = [[[100,500], [500,500]],[],[[500,500], [100,500]],[]]
    flow = Aniv_yaml(full_shot_name, detail_shot_list, key_points, 1000)

    print(flow)
    with open('yaml_output.yaml', "w", encoding="utf-8") as f:
        yaml.dump(flow, f)