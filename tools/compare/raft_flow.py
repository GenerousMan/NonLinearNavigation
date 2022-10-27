import sys
sys.path.append('./lib/RAFT/core')

import os
import cv2
import math
import time
import torch
import argparse
import skvideo.io
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from classes.video import Video

DEVICE = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args("--model lib/RAFT/models/raft-small.pth --small".split())


def resize_frame(frame, target_width=0, target_height=0):
    (height, width, channels) = frame.shape
    if target_width == 0 and target_height == 0:
        return frame
    elif target_width == 0:
        target_width = int(width * target_height / height)
    elif target_height == 0:
        target_height = int(height * target_width / width)
#     print(height, width, target_height, target_width)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def flow_extraction(flow, height_grid_num=12):
    (height, width, vector) = flow.shape
    grid_height = int(height / height_grid_num)
    grid_width = grid_height
    width_grid_num = math.ceil(width / grid_width)
    grid_width = int(width / width_grid_num)
#     print(grid_height, grid_width, height_grid_num, width_grid_num)
    flow_ext = block_reduce(flow, block_size=(grid_height, grid_width, 1), func=np.mean)
    return flow_ext

def calc_flow_data_of_videos(videos):
    start = time.time()
    total_frame_count = sum([video.frame_count for video in videos])

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        p = progressbar.ProgressBar()
        p.start(len(videos))
        for vi, video in enumerate(videos):
            p.update(vi+1)
            fps = video.fps
            frame_interval = fps * video.sample_time_interval
            frame_interval = frame_interval if frame_interval >= 1 else 1
            frame_number = 0
            next_frame_number = 0
            frame_count = 0
            prev_frame = None
            cur_frame = None
            cap = skvideo.io.vreader(video.path.replace('\\', '/'))
            for frame_number, frame in enumerate(cap):
                if prev_frame is None:
                    frame = resize_frame(frame, 0, 720)
                    prev_frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                    prev_frame = prev_frame[None].to(DEVICE)
                    flow = np.zeros((frame.shape[0], frame.shape[1], 2))
                    flow_row = resize_frame(flow, 100, 100)
                    video.frames[frame_count].flow_row = flow_row
                    flow_ext = flow_extraction(flow)
                    video.frames[frame_count].flow = flow_ext
                    frame_data = resize_frame(frame, 100, 100)
                    video.frames[frame_count].data = frame_data
                    next_frame_number += frame_interval
                    frame_count += 1
                elif frame_number == int(round(next_frame_number)):
                    frame = resize_frame(frame, 0, 720)

                    cur_frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                    cur_frame = cur_frame[None].to(DEVICE)
                    padder = InputPadder(prev_frame.shape)
                    prev_f, cur_f = padder.pad(prev_frame, cur_frame)
                    flow_low, flow_up = model(prev_f, cur_f, iters=20, test_mode=True)
                    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
                    flow_row = resize_frame(flow, 100, 100)
                    video.frames[frame_count].flow_row = flow_row
                    flow_ext = flow_extraction(flow)
                    video.frames[frame_count].flow = flow_ext
                    frame_data = resize_frame(frame, 100, 100)
                    video.frames[frame_count].data = frame_data
                    prev_frame = cur_frame
                    next_frame_number += frame_interval
                    frame_count += 1
    p.finish()
    end = time.time()
    cost = end - start
    print('RAFT time cost:', cost)
    print('Frame per second:', total_frame_count / cost)

def calc_flow_of_videos(videos):
    start = time.time()
    total_frame_count = sum([video.frame_count for video in videos])
#     temp_frame_count = 0
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        p = progressbar.ProgressBar()
        p.start(len(videos))
        for vi, video in enumerate(videos):
            p.update(vi+1)
            fps = video.fps
            frame_interval = fps * video.sample_time_interval
            frame_interval = frame_interval if frame_interval >= 1 else 1
            frame_number = 0
            next_frame_number = 0
            frame_count = 0
            prev_frame = None
            cur_frame = None
            cap = skvideo.io.vreader(video.path.replace('\\', '/'))
            for frame_number, frame in enumerate(cap):
                if prev_frame is None:
                    frame = resize_frame(frame, 0, 720)
                    prev_frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                    prev_frame = prev_frame[None].to(DEVICE)
                    flow = np.zeros((frame.shape[0], frame.shape[1], 2))
                    flow_ext = flow_extraction(flow)
                    video.frames[frame_count].flow = flow_ext
                    next_frame_number += frame_interval
                    frame_count += 1
                elif frame_number == int(round(next_frame_number)):
                    frame = resize_frame(frame, 0, 720)
#                     print(frame_number)
                    cur_frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                    cur_frame = cur_frame[None].to(DEVICE)
                    padder = InputPadder(prev_frame.shape)
                    prev_f, cur_f = padder.pad(prev_frame, cur_frame)
                    flow_low, flow_up = model(prev_f, cur_f, iters=20, test_mode=True)
                    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
                    flow_ext = flow_extraction(flow)
                    video.frames[frame_count].flow = flow_ext
                    prev_frame = cur_frame
                    next_frame_number += frame_interval
                    frame_count += 1
    p.finish()
    end = time.time()
    cost = end - start
    print('RAFT time cost:', cost)
    print('Frame per second:', total_frame_count / cost)


def calc_flow_row_of_videos(videos):
    start = time.time()
    total_frame_count = sum([video.frame_count for video in videos])
#     temp_frame_count = 0
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        p = progressbar.ProgressBar()
        p.start(len(videos))
        for vi, video in enumerate(videos):
            p.update(vi+1)
            fps = video.fps
            frame_interval = fps * video.sample_time_interval
            frame_interval = frame_interval if frame_interval >= 1 else 1
            frame_number = 0
            next_frame_number = 0
            frame_count = 0
            prev_frame = None
            cur_frame = None
            cap = skvideo.io.vreader(video.path.replace('\\', '/'))
            for frame_number, frame in enumerate(cap):
                if prev_frame is None:
                    frame = resize_frame(frame, 0, 720)
                    prev_frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                    prev_frame = prev_frame[None].to(DEVICE)
                    flow = np.zeros((100, 100, 2))
#                     flow_ext = flow_extraction(flow)
                    video.frames[frame_count].flow_row = flow
                    next_frame_number += frame_interval
                    frame_count += 1
                elif frame_number == int(round(next_frame_number)):
                    frame = resize_frame(frame, 0, 720)
#                     print(frame_number)
                    cur_frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                    cur_frame = cur_frame[None].to(DEVICE)
                    padder = InputPadder(prev_frame.shape)
                    prev_f, cur_f = padder.pad(prev_frame, cur_frame)
                    flow_low, flow_up = model(prev_f, cur_f, iters=20, test_mode=True)
                    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
#                     flow_ext = flow_extraction(flow)
                    flow_row = resize_frame(flow, 100, 100)
                    video.frames[frame_count].flow_row = flow_row
                    prev_frame = cur_frame
                    next_frame_number += frame_interval
                    frame_count += 1
    p.finish()
    end = time.time()
    cost = end - start
    print('RAFT time cost:', cost)
    print('Frame per second:', total_frame_count / cost)