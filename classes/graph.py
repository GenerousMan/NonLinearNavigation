from tools.video_features.alphapose import *
from tools.video_features.roi import *
from tools.video_features.human_features import *
from tools.video_features.combine_pose_roi import *
from tools.video_features.cloth_features import *
from tools.video_features.cloth_segmentation import *
from sklearn import preprocessing
from collections import Counter
from skimage import data,exposure

class Graph():
    def __init__(self,videos, texts, debug = False, w_list = [0.2,0.2,0.3,0.3]) -> None:
        self.video_nodes = videos
        self.text_nodes = texts

        # 三种边的记录邻接矩阵，在get_edge_relations函数中得到计算
        self.human_edge = None
        self.cloth_edge = None
        self.visual_edge = None

        self.video_nodes.sort(key = self.get_v_node_key)

        if(debug == False):
            # 先对视频节点作计算，提取特征
            self.extract_visual_feature(self.video_nodes)
            self.extract_cloth_feature(self.video_nodes)
            self.extract_human_feature(self.video_nodes)
            self.tokenize_text(self.text_nodes)

        # 再对计算特征作相似关系计算
        self.calc_vv_relation(w_list[0],w_list[1],w_list[2],w_list[3])
        self.calc_tv_relation()
        self.human_matrix = exposure.equalize_hist(self.human_matrix)
        self.cloth_matrix = exposure.equalize_hist(self.cloth_matrix)
        self.visual_matrix = exposure.equalize_hist(self.visual_matrix)


        # 再对整张图的临接关系作计算
        self.get_edge_relations()
        
    
    def get_v_node_key(self, node):
        return node.name

    def calc_tv_relation(self):
        # 计算文字、视频节点的相似度关系
        pass

    def calc_vv_relation(self,attr,cate,color,back):

        min_max_scaler = preprocessing.MinMaxScaler()
        # 计算视频和视频节点之间的相似度关系
        nodes_num = len(self.video_nodes)
        self.human_matrix = np.ones((nodes_num,nodes_num))
        self.cloth_matrix = np.ones((nodes_num,nodes_num))
        self.visual_matrix = np.ones((nodes_num,nodes_num))
        for i in range(nodes_num):
            # print(self.video_nodes[i].name)
            for j in range(i+1,nodes_num):
                # human_relation = self.calc_vv_human(self.video_nodes[i],self.video_nodes[j])
                cloth_relation = self.calc_vv_cloth(self.video_nodes[i],self.video_nodes[j],attr,cate,color,back)
                # visual_relation = self.calc_vv_visual(self.video_nodes[i],self.video_nodes[j])
                # self.human_matrix[i][j] = human_relation
                # self.human_matrix[j][i] = human_relation
                self.cloth_matrix[i][j] = cloth_relation
                self.cloth_matrix[j][i] = cloth_relation
                # self.visual_matrix[i][j] = visual_relation
                # self.visual_matrix[j][i] = visual_relation

        # self.human_matrix = min_max_scaler.fit_transform(self.human_matrix)
        # self.cloth_matrix = min_max_scaler.fit_transform(self.cloth_matrix)
        # print("human_matrix:\n", self.human_matrix)
        # print("cloth_matrix:\n", self.cloth_matrix)
    
    def calc_vv_human(self, v_node_a, v_node_b):
        view_w = 0.7
        dire_w = 0
        pose_w = 0.3
        moti_w = 0

        avg_a_view = int(0.5 + np.mean(np.array(
            [v_node_a.frames[i].view for i in range(len(v_node_a.frames))])))
        avg_a_dire = int(0.5 + np.mean(np.array(
            [v_node_a.frames[i].direction for i in range(len(v_node_a.frames))])))
        avg_a_pose = int(0.5 + np.mean(np.array(
            [v_node_a.frames[i].pose for i in range(len(v_node_a.frames))])))
        avg_a_moti = int(0.5 + np.mean(np.array(
            [v_node_a.frames[i].motion for i in range(len(v_node_a.frames))])))
        
        avg_b_view = int(0.5 + np.mean(np.array(
            [v_node_b.frames[i].view for i in range(len(v_node_b.frames))])))
        avg_b_dire = int(0.5 + np.mean(np.array(
            [v_node_b.frames[i].direction for i in range(len(v_node_b.frames))])))
        avg_b_pose = int(0.5 + np.mean(np.array(
            [v_node_b.frames[i].pose for i in range(len(v_node_b.frames))])))
        avg_b_moti = int(0.5 + np.mean(np.array(
            [v_node_b.frames[i].motion for i in range(len(v_node_b.frames))])))

        view_dis = self.view_tag_distance(avg_a_view,avg_b_view)
        dire_dis = self.direction_tag_distance(avg_a_dire,avg_b_dire)
        pose_dis = self.pose_tag_distance(avg_a_pose,avg_b_pose)
        moti_dis = self.motion_tag_distance(avg_a_moti,avg_b_moti)

        # print("human similarity:", view_dis,dire_dis,pose_dis,moti_dis)
        
        return view_dis*view_w + dire_dis*dire_w + pose_dis*pose_w + moti_dis*moti_w
        

    def calc_vv_cloth(self, v_node_a, v_node_b, attr_w, cate_w, color_w, back_w):
        cloth_attr_w = attr_w
        cloth_cate_w = cate_w
        cloth_color_w = color_w
        back_color_w = back_w


        attr_a_array = np.array([v_node_a.frames[i].cloth_attr for i in range(len(v_node_a.frames))])
        attr_b_array = np.array([v_node_b.frames[i].cloth_attr for i in range(len(v_node_b.frames))])

        avg_attr_a = np.mean(attr_a_array, axis = 0).tolist()
        avg_attr_b = np.mean(attr_b_array, axis = 0).tolist()

        cate_a_array = np.array([v_node_a.frames[i].cloth_cate for i in range(len(v_node_a.frames))])
        cate_b_array = np.array([v_node_b.frames[i].cloth_cate for i in range(len(v_node_b.frames))])

        avg_cate_a = np.mean(cate_a_array, axis = 0).tolist()
        avg_cate_b = np.mean(cate_b_array, axis = 0).tolist()


        color_a_array = np.array([(v_node_a.frames[i].cloth_max_color).reshape(3) for i in range(len(v_node_a.frames))])
        color_b_array = np.array([(v_node_b.frames[i].cloth_max_color).reshape(3) for i in range(len(v_node_b.frames))])

        color_a = self.select_color(color_a_array)
        color_b = self.select_color(color_b_array)        
        
        back_a_array = np.array([(v_node_a.frames[i].back_color).reshape(3) for i in range(len(v_node_a.frames))])
        back_b_array = np.array([(v_node_b.frames[i].back_color).reshape(3) for i in range(len(v_node_b.frames))])

        # avg_back_a = np.mean(back_a_array, axis = 0).tolist()
        # avg_back_b = np.mean(back_b_array, axis = 0).tolist()
        back_a = self.select_color(back_a_array)
        back_b = self.select_color(back_b_array)
        

        # 6类的特征计算，暂时弃用
        # color_a_array = np.array([(v_node_a.frames[i].cloth_color).reshape(18) for i in range(len(v_node_a.frames))])
        # color_b_array = np.array([(v_node_b.frames[i].cloth_color).reshape(18) for i in range(len(v_node_b.frames))])

        # avg_color_a = np.mean(color_a_array, axis = 0).tolist()
        # avg_color_b = np.mean(color_b_array, axis = 0).tolist()    

        # # print(avg_color_a)

        dis_attr = self.cosine_similarity(avg_attr_a, avg_attr_b)
        dis_cate = self.cosine_similarity(avg_cate_a, avg_cate_b)
        dis_color = self.cosine_similarity(color_a, color_b)
        dis_back = self.cosine_similarity(back_a, back_b)


        # print("cloth similarity:", dis_attr, dis_cate, dis_color, dis_back)

        return dis_attr*cloth_attr_w + dis_cate*cloth_cate_w  + dis_color*cloth_color_w +dis_back*back_color_w

    def select_color(self, color_array):
        new_color_list = []
        for i in range(color_array.shape[0]):
            color_now = color_array[i]
            # print(color_now)
            color_now_hash = [int(color_now[j]/15) for j in range(color_now.shape[0])]
            # print(color_now_hash)
            # 将一定范围内的颜色组合为同样的string，以便用counter进行计数
            color_now_str = str(color_now_hash[0])+"-"+str(color_now_hash[1])+"-"+str(color_now_hash[2])
            new_color_list.append(color_now_str)
        most_color = Counter(new_color_list)
        most_color.pop("0-0-0")
        # print(most_color)
        if(len(most_color.keys())==0):
            return [0,0,0]

        most_color = most_color.most_common(1)[0][0]
        # print(most_color)
        color_return = most_color.split("-")
        color_return = [(int(color_return[i])-0.5)*15 for i in range(len(color_return))]

        return color_return



    def calc_vv_visual(self,v_node_a, v_node_b):
        cloth_color_w = 0.6
        back_color_w = 0.4

        color_a_array = np.array([(v_node_a.frames[i].cloth_max_color).reshape(3) for i in range(len(v_node_a.frames))])
        color_b_array = np.array([(v_node_b.frames[i].cloth_max_color).reshape(3) for i in range(len(v_node_b.frames))])

        # print("a:",color_a_array)
        # print("b:",color_b_array)

        # avg_color_a = np.mean(color_a_array, axis = 0).tolist()
        # avg_color_b = np.mean(color_b_array, axis = 0).tolist()
        color_a = self.select_color(color_a_array)
        color_b = self.select_color(color_b_array)        
        
        back_a_array = np.array([(v_node_a.frames[i].back_color).reshape(3) for i in range(len(v_node_a.frames))])
        back_b_array = np.array([(v_node_b.frames[i].back_color).reshape(3) for i in range(len(v_node_b.frames))])

        # avg_back_a = np.mean(back_a_array, axis = 0).tolist()
        # avg_back_b = np.mean(back_b_array, axis = 0).tolist()
        back_a = self.select_color(back_a_array)
        back_b = self.select_color(back_b_array)
        
        dis_color = self.euclidean_similarity(color_a, color_b)
        dis_back = self.euclidean_similarity(back_a, back_b)

        return dis_color*cloth_color_w + dis_back*back_color_w


    def euclidean_similarity(self, x, y):
        x= np.array(x)
        y = np.array(y)

        return 1 - np.sqrt(np.sum(np.square(x-y)))

    def cosine_similarity(self, x, y, norm=True):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)

        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


    def tokenize_text(self, text_nodes):
        # 文字信息的抽取
        pass

    def extract_visual_feature(self, video_nodes):
        # 画面视觉特征抽取，目前使用的还是服装分割的特征。
        # 暂定：色彩联通度等画面丰富度的计算。

        pass


    def extract_cloth_feature(self, video_nodes):
        # 服装特征抽取
        calc_cloth_of_videos(video_nodes)
        calc_seg_of_videos(video_nodes)
        pass

    def extract_human_feature(self, video_nodes):
        # 人物特征抽取
        calc_pose_of_videos_v2(video_nodes)
        calc_roi_of_videos(video_nodes)
        combine_roi_pose(video_nodes)
        calc_features(video_nodes)

        pass

    def get_edge_relations(self):
        pass

    @staticmethod
    def view_tag_distance(tag1, tag2):
        # {"full-shot":0, "whole-body":1, "above-knee":2, "upper-body":3, "lower-body":4,
        # "upper-cloth":5, "portrait":6, "waist":7, "detail":8, "scene":9}
        distance = [
            [0, 1, 2, 3, 2, 4, 4, 3, 5, 6],
            [1, 0, 1, 2, 1, 3, 3, 2, 4, 6],
            [2, 1, 0, 1, 2, 2, 2, 1, 3, 6],
            [3, 2, 1, 0, 3, 1, 1, 2, 2, 6],
            [2, 1, 2, 3, 0, 4, 4, 3, 1, 6],
            [4, 3, 2, 1, 4, 0, 2, 3, 1, 6],
            [4, 3, 2, 1, 4, 2, 0, 3, 1, 6],
            [3, 2, 1, 2, 3, 3, 3, 0, 1, 6],
            [5, 4, 3, 2, 1, 1, 1, 1, 0, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 0]
        ]
        max_disance = 6
        return 1-(distance[tag1][tag2] / max_disance)

    @staticmethod
    def direction_tag_distance(tag1, tag2):
        # {"left":0, "half-left":1, "center":2, "half-right":3, "right":4, "back":5, "none":6}
        distance = [
            [0, 1, 2, 3, 4, 1, 5],
            [1, 0, 1, 2, 3, 2, 5],
            [2, 1, 0, 1, 2, 3, 5],
            [3, 2, 1, 0, 1, 2, 5],
            [4, 3, 2, 1, 0, 1, 5],
            [1, 2, 3, 2, 1, 0, 5],
            [5, 5, 5, 5, 5, 5, 0]
        ]
        max_disance = 5
        return 1-(distance[tag1][tag2] / max_disance)

    @staticmethod
    def pose_tag_distance(tag1, tag2):
        # {'stand': 0,'sit': 1,'walk': 2,'spin': 3,'none': 4 }
        distance = [
            [0, 2, 1, 1, 3],
            [2, 0, 2, 2, 3],
            [1, 2, 0, 1, 3],
            [1, 2, 1, 0, 3],
            [1, 3, 3, 1, 0]
        ]
        max_disance = 3
        return 1-(distance[tag1][tag2] / max_disance)

    @staticmethod
    def motion_tag_distance(tag1, tag2):
        # {'still': 0,'low': 1,'high': 2,'none': 3}
        distance = [
            [0, 1, 2, 3],
            [1, 0, 1, 3],
            [2, 1, 0, 3],
            [3, 3, 3, 0],
        ]
        max_disance = 3
        return 1-(distance[tag1][tag2] / max_disance)