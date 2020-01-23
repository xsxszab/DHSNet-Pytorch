
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_mae(pred, gt, in_type='numpy'):
    """Get average MAE for evaluation.

    :param pred: predicted saliency map, if in_type='file', pred is the path to images,
    if in_type='numpy' it should be a 3D ndarray having dtype np.float and of shape [length, height, width].
    :param gt: ground truth saliency map, if in_type='file', gt is the path to images
    if in_type='numpy' it should be a 3D ndarray having dtype np.uint8 and of shape [length, height, width].
    :param in_type: value in {'file', 'numpy'}, default set to 'numpy'.
    :return: one float value, indicating average MAE for evaluation.
    """
    if in_type == 'file':
        pred_names = os.listdir(pred)
        pred_list = []
        for name in pred_names:
            img = np.array(Image.open(name, 'r'))
            img = img.astype(np.float)
            pred_list.append(img)
        pred_list = np.array(pred_list)

        gt_names = os.listdir(gt)
        gt_list = []
        for name in gt_names:
            gt = np.array(Image.open(name, 'r'))
            gt = img.astype(np.float)
            gt_list.append(gt)
        gt_list = np.array(gt_list)

    elif in_type == 'numpy':
        assert pred.shape == gt.shape
        pred_list = pred
        gt_list = gt

    else:
        raise ValueError('invalid in_type')

    mae = np.mean(
        np.absolute(
            pred_list.astype("float") -
            gt_list.astype("float")))  # calculate mean absolute error
    return mae


def get_f_measure(pred, gt, in_type='numpy', beta_square=0.3, threshold='adaptive'):
    """Get average F measurefor evaluation.

    :param pred: predicted saliency map, if in_type='file', pred is the path to images,
    if in_type='numpy' it should be a 3D ndarray having dtype np.float and of shape [length, height, width].
    :param gt: ground truth saliency map, if in_type='file', gt is the path to images
    if in_type='numpy' it should be a 3D ndarray having dtype np.uint8 and of shape [length, height, width].
    :param in_type: value in {'file', 'numpy'}, default set to 'numpy'.
    :param beta_square: value for $beta^2$ in f measure calculation, default set to 0.3.
    :param threshold: 'adaptive' or 1D ndarray of float value(not recommended), default set to 'adaptive',
    which means threshold will be set to twice the mean saliency value of each saliency map.
    :return: one float value, indicating average F measure.
    """
    if in_type == 'file':  # if in_type is file, convert images to ndarray
        pred_names = os.listdir(pred)
        pred_list = []
        for name in pred_names:
            img = np.array(Image.open(name, 'r'))
            img = img.astype(np.float)
            pred_list.append(img)
        pred_list = np.array(pred_list)

        gt_names = os.listdir(gt)
        gt_list = []
        for name in gt_names:
            gt = np.array(Image.open(name, 'r'))
            gt = gt.astype(np.float)
            gt_list.append(gt)
        gt_list = np.array(gt_list)

    elif in_type == 'numpy':
        assert pred.shape == gt.shape
        pred_list = pred
        gt_list = gt

    else:
        raise ValueError('invalid in_type')

    if threshold == 'adaptive':
        length = pred_list.shape[0]  # num of images
        # 1D ndarray, containing mean saliency value for each image
        mean_sal = np.mean(pred_list, (1, 2))
        # transpose to shape [height, width, length]
        pred_list = pred_list.transpose((1, 2, 0))
        # numpy propagation [height, width, length] - [length]
        pred_list -= 2 * mean_sal

    if type(threshold) is float:
        pred_list -= threshold
    eps = np.finfo(float).eps  # eps can be added to denominator to avoid dividing zero.
    gt_list = gt_list.transpose((1, 2, 0))  # transpose to shape [height, width, length]
    pred_list = pred_list.reshape([-1, length])  # flatten images to [height*width, length]
    gt_list = gt_list.reshape([-1, length])  # flatten ground truth maps to [height*width, length]
    pred_list[pred_list > 0] = 1
    pred_list[pred_list <= 0] = 0
    pred_list = pred_list.astype(np.uint8)
    TP = np.sum(np.multiply(gt_list, pred_list), 0)  # true positive rate
    FP = np.sum(np.logical_and(np.equal(gt_list, 0), np.equal(pred_list, 1)), 0)  # false positive rate
    FN = np.sum(np.logical_and(np.equal(gt_list, 1), np.equal(pred_list, 0)), 0)  # true negative rate
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f_measure = ((1 + beta_square)*precision*recall) / (beta_square*precision + recall + eps)
    f_measure = np.mean(f_measure)
    return f_measure





def saveimg(img ,save_path, name, save_type='png'):
    """Save single saliency map to disk.

    :param img: 2D ndarray, image to save.
    :param save_path: Image save path.
    :param name: Image save name.
    :param save_type: String value, indicating image save type, default set to png.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    full_name = name + '.' + save_type
    full_path = os.path.join(save_path, full_name)
    plt.imsave(full_path, img, cmap='gray')

def test():
    """test get_mae and get_f_measure"""
    np.random.seed(100)
    test_pred = np.random.rand(100, 224, 224).astype(np.float)
    # test_pred = np.ones((100, 224, 224)).astype(np.float)
    test_gt = np.ones((100, 224, 224)).astype(np.uint8)
    F_measure = get_f_measure(test_pred, test_gt, in_type='numpy')
    MAE = get_mae(test_pred, test_gt, in_type='numpy')
    print('f measure:', F_measure)
    print('MAE:', MAE)
    print('test done')

if __name__ == '__main__':
    test()
