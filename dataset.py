
import os
import time

import numpy as np
from PIL import Image
import torch
from torch.utils import data


class TrainData(data.Dataset):
    """Custom training data class.

    Loads input images and corresponding ground truth maps.
    """

    def __init__(self, path, transform=True):
        """Init Data class.

        :param path: Str, Path to dataset directory, it should contain at least
        two subdirectory <path>/raw (for input images)
        and <path>/mask (for ground truth images).
        :param transform: Boolean, Default value set to True,
        if True transform inputs by transform instance method.
        """
        super().__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        input_path = os.path.join(path, 'raw/')
        gt_path = os.path.join(path, 'mask/')
        self.transform_flag = transform
        # print('pwd:', os.getcwd())  # uncomment to test dir
        names = os.listdir(input_path)
        self.length = len(names)
        if self.length == 0:
            raise RuntimeError('training dataset is empty')
        if self.length != len(os.listdir(gt_path)):
            raise RuntimeError('cannot match raw and gt input')
        self.input_names = []  # absolute path
        self.gt_names = []  # absolute path
        for name in names:
            self.input_names.append(input_path + name[:-4] + '.jpg')
            self.gt_names.append(gt_path + name[:-4] + '.png')

    def __len__(self):
        """return length of current dataset"""
        return self.length

    def __getitem__(self, index):
        """return next raw image and corresponding ground truth map"""
        img_path = self.input_names[index]
        img = Image.open(img_path, 'r')
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.uint8)

        gt_path = self.gt_names[index]
        gt = Image.open(gt_path, 'r')
        gt = gt.resize((224, 224))
        gt = np.array(gt, dtype=np.uint32)
        gt[gt != 0] = 1

        if self.transform_flag:
            img, gt = self.transform(img, gt)
        return img, gt

    def transform(self, img, gt):
        """Standardize raw image and corresponding ground truth map,
        and convert to pytorch data format.
        """
        img = img.astype(np.float32) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        gt = gt.astype(np.float32)
        gt = torch.from_numpy(gt).float()
        return img, gt


class TestData(data.Dataset):
    """Custom testing data class,loads input images.
    """

    def __init__(self, path, transform=True):
        """Init Data class.

        :param path: Path to dataset directory, it should contain at least
        one subdirectory <path>/raw (for input images).
        :param transform: Boolean, Default value set to True,
        if True transform inputs by transform instance method.
        """
        super().__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.size = (224, 224)
        input_path = path + 'raw/'
        self.transform_flag = transform
        self.names = os.listdir(input_path)
        self.length = len(self.names)
        if self.length == 0:
            raise RuntimeError('testing dataset is empty')
        self.input_names = []  # absolute path
        for name in self.names:
            self.input_names.append(input_path + name)

    def __len__(self):
        """return length of current dataset"""
        return self.length

    def __getitem__(self, index):
        """return next raw image"""
        img_path = self.input_names[index]
        img = Image.open(img_path)
        img = img.resize(self.size)
        img = np.array(img, dtype=np.uint8)

        if self.transform_flag:
            img = self.transform(img)
        return img, self.names[index]

    def transform(self, img):
        """Standardize raw image,
        and convert to pytorch data format.
        """
        img = img.astype(np.float64) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img


def test(path):
    """Test Data class
    :param path: Input path for TrainData and TestData class
    """
    trainclass = TrainData(path=path, transform=True)
    testclass = TestData(path=path, transform=True)
    print('train data length:', len(trainclass))
    print('test data length:', len(testclass))
    print('Data class test done.')


if __name__ == '__main__':
    test('./input/train/')
