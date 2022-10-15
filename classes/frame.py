import json
import numpy as np
class Frame:

    def __init__(self, frame_number, frame_np):
        self.frame_number = frame_number
        self.frame_np = frame_np
        self.alphapose = {}
        self.roi = [0, 0, 0, 0]
        self.comb_data = []

        self.view = ''
        self.pose = ''
        self.direction = ''
        self.motion = ''

        self.cloth_cate = []
        self.cloth_attr = []
        self.cloth_color = np.zeros((6,3))
        self.cloth_max_color = np.zeros((3))
        self.back_color = np.zeros((3))


    def print_self(self):
        print("frame number:", self.frame_number)
        print("numpy shape:", self.frame_np.shape)
        print("pose:",self.alphapose)
        print("roi:",self.roi)
        print("view:",self.view)
        print("pose:",self.pose)
        print("direction:",self.direction)
        print("motion:",self.motion)

        print("cloth category:",self.cloth_cate)
        print("cloth attribute:",self.cloth_attr)
        print("cloth_color:",self.cloth_color)
        print("cloth_max_color:",self.cloth_max_color)
        print("back_color:",self.back_color)
    def __str__(self):
        return 'Frame {}'.format(frame_number)