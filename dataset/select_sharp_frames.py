import os
import numpy as np
import cv2
import argparse
import shutil
from util.util import make_dataset

parser = argparse.ArgumentParser('select sharp frames')
parser.add_argument('--data_root', type=str, default="E:/VideoDeblur/", help='path to dataset')
parser.add_argument('--number', type=int, default=20, help='the number of frames to select one sharp frame')
parser.add_argument('--sharp_root', type=str, default='E:/VideoDeblur/low_light_blur/low_light_sharp', help='output directory')
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
            image_VL_patch_max = np.zeros(4)   # the max variance of the image Laplacian
            sharp_patch_frame = np.zeros(4)
            i = 1
            for idx in range(index, index + args.number):
                data_path = data_paths[idx]
                image = cv2.imread(data_path)
                h, w, _ = image.shape
                patch1 = image[0:h//2, 0:w//2, :]
                patch2 = image[0:h//2, w//2:w, :]
                patch3 = image[h//2:h, 0:w//2, :]
                patch4 = image[h//2:h, w//2:w, :]
                image_VL_patch1 = calculate_VL(patch1)
                image_VL_patch2 = calculate_VL(patch2)
                image_VL_patch3 = calculate_VL(patch3)
                image_VL_patch4 = calculate_VL(patch4)
                if image_VL_patch_max[0] <= image_VL_patch1:
                    image_VL_patch_max[0] = image_VL_patch1
                    sharp_patch_frame[0] = idx
                if image_VL_patch_max[1] <= image_VL_patch2:
                    image_VL_patch_max[1] = image_VL_patch2
                    sharp_patch_frame[1] = idx
                if image_VL_patch_max[2] <= image_VL_patch3:
                    image_VL_patch_max[2] = image_VL_patch3
                    sharp_patch_frame[2] = idx
                if image_VL_patch_max[3] <= image_VL_patch4:
                    image_VL_patch_max[3] = image_VL_patch4
                    sharp_patch_frame[3] = idx
            for idx in sharp_patch_frame:
                patch_path = data_paths[index + int(idx)]
                print(patch_path)
                patch_name = os.path.splitext(os.path.split(patch_path)[1])[0]
                patch_name = '%s_patch%d.png' % (patch_name, i)
                sharp_patch = cv2.imread(patch_path)
                h, w, _ = sharp_patch.shape
                if i == 1:
                    sharp_patch = cv2.imread(patch_path)[0:h//2, 0:w//2, :]
                elif i == 2:
                    sharp_patch = cv2.imread(patch_path)[0:h//2, w//2:w, :]
                elif i == 3:
                    sharp_patch = cv2.imread(patch_path)[h//2:h, 0:w//2, :]
                else:
                    sharp_patch = cv2.imread(patch_path)[h//2:h, w//2:w, :]
                cv2.imwrite(os.path.join(args.sharp_root, patch_name), sharp_patch)
                i += 1
            # remove_file(data_paths[sharp_frame], args.sharp_root)
            print(image_VL_patch_max)
            index += args.number

if __name__ == '__main__':
    main()