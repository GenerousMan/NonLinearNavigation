# from classes.vshot_group import VShotGroup
from classes.vshot_v2 import VShot
import random

class Editor:

    def __init__(self) :
        self.vshot_groups = []
        # self.last_pair = None
        self.curr_pair = None
        self.select_view = 1 # full-shot
        self.select_direct = 2 # center
        self.select = 'view'

    def add_vshot_group(self, vshot_group):
        vshot_group.reset_all_vshots()
        self.vshot_groups.append(vshot_group)
        # TODO 添加商品后要挑一个视频加入当前镜头组进行播放
        if self.curr_pair is None:
            add_vshot = self.get_match_vshot(vshot_group)
            self.curr_pair = [add_vshot]
            vshot_group.set_unplayable_vshots(add_vshot)
        else:
            add_vshot = self.get_match_vshot(vshot_group)
            if add_vshot is not None:
                self.curr_pair.append(add_vshot)
                vshot_group.set_unplayable_vshots(add_vshot)
            else:
                print('No match vshot can add to curr!')
                self.curr_pair.append(None)

    def get_match_vshot(self, vshot_group):
        vshots = vshot_group.get_playable_vshots()
        if self.curr_pair is not None:
            vshots = [vshot for vshot in vshots if vshot.len == self.curr_pair[0].len]
        vshots = [vshot for vshot in vshots if self.get_vshot_feature_match(vshot)]
        if len(vshots) == 0:
            vshots = vshot_group.get_playable_vshots()
            if self.curr_pair is not None:
                vshots = [vshot for vshot in vshots if vshot.len == self.curr_pair[0].len]
        if len(vshots) == 0:
            print('No pair match! Bad videos here!')
            return None
        vshots = sorted(vshots, key=lambda vshot:self.get_vshot_feature_loss(vshot) / vshot.len)
        return vshots[0]

    def get_vshot_feature_loss(self, vshot):
        if self.select == 'view':
            return abs(vshot.view_mean - self.select_view)
        elif self.select == 'direct':
            return abs(vshot.direct_mean - self.select_direct)
        else:
            return 0

    def del_vshot_group(self, index):
        del self.vshot_groups[index]
        # TODO 删除商品后要在当前组里也删除
        if self.curr_pair is not None:
            del self.curr_pair[index]
        if len(self.vshot_groups) == 0:
            self.curr_pair = None
            self.select = 'view'
            self.select_view = 1 # full-shot
            self.select_direct = 2 # center

    def get_feature_accpet_vshots_list(self):
        vshots_list = []
        for vsg, cvshot in zip(self.vshot_groups, self.curr_pair):
            vshots = []
            while len(vshots) < 1:
                vshots = vsg.get_playable_vshots()
                vshots = [vshot for vshot in vshots if not cvshot.cross_with(vshot, 16)]
                vshots = [vshot for vshot in vshots if self.get_vshot_feature_match(vshot)]
                if len(vshots) >= 1:
                    break
                elif len(vsg.played_vshots) == 0:
                    vshots = vsg.get_playable_vshots()
                    vshots = [vshot for vshot in vshots if not cvshot.cross_with(vshot, 16)]
                    break
                else:
                    vsg.reset_farthest_vshot()
            # 如果列表太多了 随机挑一些
            if len(vshots) > 50:
                print(len(vshots))
                vshots = random.sample(vshots, 50)
            vshots_list.append(vshots)
        return vshots_list

    def get_vshots_list(self):
        vshots_list = []
        for vsg, cvshot in zip(self.vshot_groups, self.curr_pair):
            vshots = []
            while len(vshots) < 1:
                vshots = vsg.get_playable_vshots()
                vshots = [vshot for vshot in vshots if not cvshot.cross_with(vshot, 16)]
                # vshots = [vshot for vshot in vshots if self.get_vshot_feature_match(vshot)]
                if len(vshots) >= 1:
                    break
                elif len(vsg.played_vshots) == 0:
                    vshots = vsg.get_playable_vshots()
                    vshots = [vshot for vshot in vshots if not cvshot.cross_with(vshot, 16)]
                    break
                else:
                    vsg.reset_farthest_vshot()
            # 如果列表太多了 随机挑一些
            if len(vshots) > 50:
                # print(len(vshots))
                vshots = random.sample(vshots, 50)
            vshots_list.append(vshots)
        return vshots_list

    def get_feature_accpet_pairs(self, vshots_list):
        last_pairs = None
        new_pairs = []
        for i in range(len(vshots_list)):
            if last_pairs == None:
                for vshot in vshots_list[i]:
                    if self.select == 'view' and vshot.view_match(self.select_view):
                        new_pairs.append([vshot])
                    elif self.select == 'direct' and vshot.direct_match(self.select_direct):
                        new_pairs.append([vshot])
            else:
                for vshot in vshots_list[i]:
                    for pair in last_pairs:
                        accpet = True
                        for pvshot in pair:
                            if not accpet:
                                break
                            elif pvshot.len != vshot.len:
                                accpet = False
                            elif self.select == 'view' and not vshot.view_match(self.select_view):
                                accpet = False
                            elif self.select == 'direct' and not vshot.direct_match(self.select_direct):
                                accpet = False
                        if accpet:
                            new_pair = pair.copy()
                            new_pair.append(vshot)
                            new_pairs.append(new_pair)
            last_pairs = new_pairs
            new_pairs = []
        return last_pairs

    def get_feature_approx_pairs(self, vshots_list):
        last_pairs = None
        new_pairs = []
        for i in range(len(vshots_list)):
            if last_pairs == None:
                for vshot in vshots_list[i]:
                    new_pairs.append([vshot])
            else:
                for vshot in vshots_list[i]:
                    for pair in last_pairs:
                        accpet = True
                        for pvshot in pair:
                            if not accpet:
                                break
                            elif pvshot.len != vshot.len:
                                accpet = False
                        if accpet:
                            new_pair = pair.copy()
                            new_pair.append(vshot)
                            new_pairs.append(new_pair)
            last_pairs = new_pairs
            new_pairs = []
        return last_pairs

    def get_vshot_feature_match(self, vshot):
        if self.select == 'view' and vshot.view_match(self.select_view):
            return True
        elif self.select == 'direct' and vshot.direct_match(self.select_direct):
            return True
        else:
            return False

    def get_pair_feature_score(self, pair):
        score = 0.0
        for vshot in pair:
            if self.select == 'view':
                score += 1 - abs(vshot.view_mean - self.select_view)
            elif self.select == 'direct':
                score += 1 - abs(vshot.direct_mean - self.select_direct)
        return score

    def get_pair_seconde_feature_score(self, pair):
        # loss = 0.0
        score = 0.0
        for ivshot in pair:
            for jvshot in pair:
                if ivshot == jvshot:
                    continue
                if self.select == 'view':
                    # loss += ivshot.vshot_direct_match(jvshot)
                    score += 1 - ivshot.vshot_direct_match(jvshot)
                elif self.select == 'direct':
                    # loss += ivshot.vshot_view_match(jvshot)
                    score += 1 - ivshot.vshot_view_match(jvshot)
        return score

    def get_pair_cut_feature_loss(self, pair):
        loss = 0.0
        for vsg, curr_vshot, vshot in zip(self.vshot_groups, self.curr_pair, pair):
            if curr_vshot is not None:
                loss += vsg.cut_scores[curr_vshot.vi][vshot.vi]
        return loss

    def get_pair_flow_feature_loss(self, pair):
        loss = 0.0
        for ivshot in pair:
            for jvshot in pair:
                if ivshot == jvshot:
                    continue
                loss += ivshot.calc_flow_diff(jvshot)
        return loss

    def get_accept_pairs(self):
        min_len_limit = 16
        # 看看目前能够出多少个满足需求的对
        # vshots_list = self.get_feature_accpet_vshots_list()
        vshots_list = self.get_vshots_list()
        # 每一个vshots_list里面至少有1个vshot
        pairs = self.get_feature_accpet_pairs(vshots_list)
        while len(pairs) < 1:
            # 没有找到，就要找一个vshots_group试探性的把已经放过的再放回去
            # 这个vshots_group找的规则，可以是总素材已播放占比最多的
            min_per = 1.1
            min_per_vsg = None
            for vsg in self.vshot_groups:
                if vsg.get_playable_percent() < min_per:
                    min_per = vsg.get_playable_percent()
                    min_per_vsg = vsg
            if len(min_per_vsg.played_vshots) > 0:
                # 如果还有可以释放的，那就释放一个
                min_per_vsg.reset_farthest_vshot()
                # vshots_list = self.get_feature_accpet_vshots_list()
                vshots_list = self.get_vshots_list()
                pairs = self.get_feature_accpet_pairs(vshots_list)
            else:
                vshots_list = self.get_vshots_list()
                # print([len(vstlist) for vstlist in vshots_list])
                # print([vshot in vstlist for vshot, vstlist in zip(self.curr_pair, vshots_list)])
                # 如果实在找不到了，那就硬着头皮上近似的特征镜头了
                pairs = self.get_feature_approx_pairs(vshots_list)
        if len(pairs) > min_len_limit:
            # pairs = sorted(pairs, key=lambda pair:self.get_pair_feature_loss(pair) / pair[0].len)
            pairs = sorted(pairs, key=lambda pair:self.get_pair_feature_score(pair) * pair[0].len, reverse=True)
            pairs = pairs[0:min_len_limit]
        # 这里至少会确保返回1个pair，没有的话就是数据集有问题了
        if len(pairs) == 0:
            print('No pair match! Bad videos here!')
        return pairs

    def get_next_pair(self):
        lens = []
        # 这里至少会保证1个镜头组
        pairs = self.get_accept_pairs()
        lens.append(len(pairs))
        # 进行第二特征排序
        if len(pairs) >= 2:
            # pairs = sorted(pairs, key=lambda pair:self.get_pair_seconde_feature_loss(pair) / pair[0].len)
            pairs = sorted(pairs, key=lambda pair:self.get_pair_seconde_feature_score(pair) * pair[0].len, reverse=True)
            pairs = pairs[0:len(pairs)//2]
        lens.append(len(pairs))
        # 进行画面相似度排序
        if len(pairs) >= 2:
            pairs = sorted(pairs, key=lambda pair:self.get_pair_flow_feature_loss(pair) / pair[0].len)
            pairs = pairs[0:len(pairs)//2]
        lens.append(len(pairs))
        # 进行镜头选择排序
        if self.curr_pair is not None and len(pairs) >= 2:
            pairs = sorted(pairs, key=lambda pair:self.get_pair_cut_feature_loss(pair), reverse=True)
            pairs = pairs[0:len(pairs)//2]
        lens.append(len(pairs))
        # print("--------------")
        # print(lens)
        if len(pairs) >= 1:
            for vsg, vshot in zip(self.vshot_groups, pairs[0]):
                vsg.set_unplayable_vshots(vshot)
            self.curr_pair = pairs[0]
            return pairs[0]
        else:
            print('No pair match! Bad videos here!')
            return None

    def change_select(self, select, value):
        assert select in ['view', 'direct']
        # 设置当前选择属性
        self.select = select
        if select == 'view':
            self.select_view = value
        elif select == 'direct':
            self.select_direct = value
        # 重置各个vshot_group
        for vsg in self.vshot_groups:
            vsg.reset_all_vshots()
        self.get_next_pair()

    def print_curr_pair_state(self):
        # 各个组的使用率占比
        print("--------------")
        for vi, vsg in enumerate(self.vshot_groups):
            print("[{}] {} {}/{}".format(vi, round(vsg.get_playable_percent(), 3), len(vsg.played_vshots), len(vsg.vshots)))
        print("--------------")
        for vi, vshot in enumerate(self.curr_pair):
            print("[{}]".format(vi), vshot)