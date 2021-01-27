import os
import numpy as np
import cv2
import argparse
import shutil
from util.util import make_dataset

parser = argparse.ArgumentParser('select sharp frames')
parser.add_argument('--data_root', type=str, default="E:/GoPro_Large/test/video_task/", help='path to dataset')
parser.add_argument('--number', type=int, default=20, help='the number of frames to select one sharp frame')
parser.add_argument('--sharp_root', type=str, default='E:/GoPro_Large/sharp/', help='output directory')
args = parser.parse_args()

# print arguments
for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

Laplacian_mask = np.mat([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

def calculate_VL(image: np.ndarray):
    """
    Calculate the variance of the image Laplacian
    :param image: the input image
    :return: the variance
    """
    image_Laplacian = cv2.filter2D(image, -1, Laplacian_mask)
    image_VL = np.var(image_Laplacian)
    return image_VL

def remove_file(old_path, new_path):
    """
    remove sharp frame to new file
    """
    assert os.path.isdir(new_path), '%s is not a valid directory' % new_path
    name = os.path.split(old_path)
    sequence = os.path.split(name[0])[1]
    new_path = os.path.join(new_path, sequence)

    if not os.path.exists(new_path):
        os.makedirs(new_path)
    shutil.move(old_path, new_path)

def main():
    """
    select sharp frames from blur video sequence
    """
    sequences = os.listdir(args.data_root)
    for sequence in sequences:
        data_paths = sorted(make_dataset(os.path.join(args.data_root, sequence)))
        index = 0
        while (index + args.number) < len(data_paths):
            image_VLmax = 0   # the max variance of the image Laplacian
            sharp_frame = 0
            for idx in range(index, index + args.number):
                data_path = data_paths[idx]
                image = cv2.imread(data_path)
                image_VL = calculate_VL(image)
                if image_VLmax <= image_VL:
                    image_VLmax = image_VL
                    sharp_frame = idx
            remove_file(data_paths[sharp_frame], args.sharp_root)
            index += args.number

if __name__ == '__main__':
    main()