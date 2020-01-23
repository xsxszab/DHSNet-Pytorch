
import os
import argparse  # parse input arguments

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from model.DHS_model import RCL  # RCL module class
from model.DHS_model import DHS  # DHS net class
from dataset import TestData
from utils.tools import saveimg


def main(args):
    """Function for inferencing saliency maps using DHS net."""
    img_dir = args.img_dir
    output_dir = args.output_dir
    model_dir = args.output_dir
    flag = (args.device == 'gpu')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = DHS(RCL)
    if flag:
        model.cuda()
    params = model_dir + 'best_network.pth'
    model.load_state_dict(torch.load(params))
    loader = torch.utils.data.DataLoader(
        TestData(img_dir),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    for (img, name) in loader:
        img = Variable(img)  # CPU version
        _, _, _, _, output = model.forward(img)
        out = output.data.cpu().numpy()
        out = np.squeeze(out)
        saveimg(out, output_dir, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DHS inference')
    parser.add_argument('--img_dir', default='./inference/', type=str)
    parser.add_argument('--output_dir', default='./output/', type=str)
    parser.add_argument('--model_dir', default='./checkpoint/', type=str)
    parser.add_argument('-d', '--device', default='cpu', type=str)
    args = parser.parse_args()
    main(args)