

class Path(object):

    def __init__(self, vshot, vshot_index):
        self.vshot_index = [vshot_index]
        self.vshot_name = [vshot.video.name]
        self.vshot_area = [(vshot.start, vshot.end)]
        self.simi_sum = vshot.simi
        self.size = 1

    def add_vshot(self, vshot, vshot_index):
        self.vshot_index.append(vshot_index)
        self.vshot_name.append(vshot.video.name)
        self.vshot_area.append((vshot.start, vshot.end))
        self.simi_sum += vshot.simi
        self.size += 1

    def get_test_score(self, vshot):
        overlap = False
        for (name, (start, end)) in zip(self.vshot_name, self.vshot_area):
            if vshot.video.name == name:
                if start >= vshot.end or vshot.start >= end:
                    continue
                else:
                    overlap = True
                    break
        if overlap:
            score = 0
        else:
            size = self.size + 1
            score = (self.simi_sum + vshot.simi) / size
        return score

    def get_score(self):
        return self.simi_sum / self.size
