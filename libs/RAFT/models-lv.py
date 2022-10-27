import sys
sys.path.append('core')

import torch
from raft import RAFT
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    DEVICE = 'cuda'
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model_name = os.path.split(args.model)[-1]
    model_lv_path = os.path.join('models-lv', model_name)

    torch.save(model.state_dict(), model_lv_path, _use_new_zipfile_serialization=False)