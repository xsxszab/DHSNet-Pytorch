
import os
import argparse  # parse input arguments

import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter  # use tensorboard for visualization
from tqdm import tqdm  # progess bar
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from model.DHS_model import DHS_network  # DHS network
from dataset import TrainData
from utils.tools import get_mae  # get MAE
from utils.tools import get_f_measure  # get adaptive f measure



def main(args):
    """main function for training DHS net"""

    # print(args) # uncomment to test arg inputs
    bsize = args.batch_size
    train_dir = args.train_dir
    test_dir = args.test_dir
    model_dir = args.ckpt_dir
    tensorboard_dir = args.tensorboard_dir
    device = args.device
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train_loader = torch.utils.data.DataLoader(
        TrainData(train_dir, transform=True),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TrainData(test_dir, transform=True),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

    model = DHS_network
    if device == 'gpu':
        model.cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss = []
    evaluation = []
    result = {'epoch': [], 'F_measure': [], 'MAE': []}

    progress = tqdm(
        range(0, args.epochs + 1),
        miniters=1,
        ncols=100,
        desc='Overall Progress',
        leave=True,
        position=0)
    offset = 1
    best = 0

    writer = SummaryWriter(tensorboard_dir)

    for epoch in progress:
        if epoch != 0:
            print("load parameters")
            model.load_state_dict(
                torch.load(model_dir + 'current_network.pth'))
            optimizer.load_state_dict(
                torch.load(model_dir + 'current_optimizer.pth'))
        title = 'Training Epoch {}'.format(epoch)
        progress_epoch = tqdm(train_loader, ncols=120,
                              total=len(train_loader), smoothing=0.9,
                              miniters=1,
                              leave=True, position=offset, desc=title)

        for ib, (img, gt) in enumerate(progress_epoch):
            # inputs = Variable(img).cuda()  # GPU version
            # gt = Variable(gt.unsqueeze(1)).cuda()  # GPU version
            inputs = Variable(img)  # CPU version
            gt = Variable(gt.unsqueeze(1))  # CPU version
            gt_28 = functional.interpolate(gt, size=28, mode='bilinear')
            gt_56 = functional.interpolate(gt, size=56, mode='bilinear')
            gt_112 = functional.interpolate(gt, size=112, mode='bilinear')
            msk1, msk2, msk3, msk4, msk5 = model.forward(inputs)

            loss = criterion(msk1, gt_28) + criterion(msk2, gt_28) +\
                criterion(msk3, gt_56) + criterion(msk4, gt_112) +\
                criterion(msk5, gt)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(round(float(loss.data.cpu()), 3))
            title = '{} Epoch {}/{}'.format('Training',
                                            epoch, args.epochs)
            progress_epoch.set_description(
                title + ' ' + 'loss:' + str(loss.data.cpu().numpy()))
            writer.add_scalar('Train/Loss', loss.data.cpu(), epoch)

        filename = model_dir + 'current_network.pth'
        filename_opti = model_dir + 'current_optimizer.pth'
        torch.save(model.state_dict(), filename)  # save current model params
        torch.save(optimizer.state_dict(), filename_opti)  # save current optimizer params

        if epoch % args.val_rate == 0:  # start validation
            params = model_dir + 'current_network.pth'
            model.load_state_dict(torch.load(params))
            pred_list = []
            gt_list = []
            for img, gt in val_loader:
                # inputs = Variable(img).cuda()  # GPU version
                inputs = Variable(img)  # CPU version
                _, _, _, _, output = model.forward(inputs)
                out = output.data.cpu().numpy()
                pred_list.extend(out)
                gt = gt.numpy()
                gt_list.extend(gt)
            pred_list = np.array(pred_list)
            pred_list = np.squeeze(pred_list)
            gt_list = np.array(gt_list)
            F_measure = get_f_measure(pred_list, gt_list)
            mae = get_mae(pred_list, gt_list)
            evaluation.append([int(epoch), float(F_measure), float(mae)])
            result['epoch'].append(int(epoch))
            result['F_measure'].append(round(float(F_measure), 3))
            result['MAE'].append(round(float(mae), 3))
            df = pd.DataFrame(result).set_index('epoch')
            df.to_csv('./eval.csv')

            if epoch == 0:
                best = F_measure - mae
            elif F_measure - mae > best:  # save model with best performance
                best = F_measure - mae
                filename = ('%s/best_network.pth' % model_dir)
                filename_opti = ('%s/best_optimizer.pth' % model_dir)
                torch.save(model.state_dict(), filename)
                torch.save(optimizer.state_dict(), filename_opti)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-5)  # set learning rate

    parser.add_argument('--train_dir', default='./input/train/', type=str)  # train data directory
    parser.add_argument('--test_dir', default='./input/test/', type=str)  # test data directory
    parser.add_argument('--ckpt_dir', default='./checkpoint/', type=str)  # trained model directory
    parser.add_argument('--tensorboard_dir', default='./tensorboard/', type=str)  # tensorboard summary directory

    parser.add_argument('--val_rate', default=4)

    parser.add_argument('-d', '--device', default='cpu', type=str)  # device to use, CPU or GPU

    args = parser.parse_args()
    main(args)
