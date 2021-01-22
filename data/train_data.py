from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from PIL import Image
from util.util import make_dataset
import torchvision.transforms as transforms
import torch.nn as nn
import tensorflow as tf

class DeblurDatset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_paths = sorted(make_dataset(opt.data_root))
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        self.gamma = 2.2
    def __getitem__(self, index):
        """
        import the images from the data root, then crop them into 256 × 256, random
        blur kernel from kernel set, and conv blur kernel with patch
        """
        data_path = self.data_paths[index]
        image = Image.open(data_path).convert('RGB')
        image = self.transform(image)
        w = image.size(2)
        h = image.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        patch = image[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        _im = tf.sign(patch) * (tf.abs(patch)) ** self.gamma
        # TODO kernel library
        kernel = []
        B, H, W, C = patch.get_shape()
        c1 = tf.nn.conv2d(_im[:, :, :, 0:1], kernel, strides=[1, 1, 1, 1], padding='SAME')
        c2 = tf.nn.conv2d(_im[:, :, :, 1:2], kernel, strides=[1, 1, 1, 1], padding='SAME')
        c3 = tf.nn.conv2d(_im[:, :, :, 2:3], kernel, strides=[1, 1, 1, 1], padding='SAME')
        result = tf.concat([c1, c2, c3], axis=3)
        patch = tf.sign(result) * (tf.abs(result)) ** (1 / self.gamma)

        return {'sharp': patch, 'sharp_paths': data_path}
    def __len__(self):
        return len(self.data_paths)
