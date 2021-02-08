from .base_dataset import BaseDataset
import numpy as np
from util.util import make_dataset
import cv2
from dataset.generate_kernel import generate_kernel_trajectory
from scipy import signal
import albumentations as albu
import copy
from dataset.select_sharp_frames import calculate_VL

class TrainDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_paths = sorted(make_dataset(opt.data_root))
        self.gamma = 2.2
        self.kernel_size = opt.kernel_size
        self.kernel = np.load(opt.kernel_path).item()

    def __getitem__(self, index):
        """
        import the images from the data root, then crop them into 256 × 256, random
        blur kernel from kernel set, and conv blur kernel with patch
        """
        data_path = self.data_paths[index]
        image = cv2.imread(data_path)
        sharp_patch = albu.RandomCrop(self.opt.fineSize, self.opt.fineSize, always_apply=True)(image=image)['image']
        sharp_patch_VL = calculate_VL(sharp_patch)
        sharp_patch = cv2.normalize(sharp_patch, sharp_patch, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        init_angle = np.math.floor(np.random.uniform(0, 180))
        init_length = np.math.floor(np.random.uniform(0, self.kernel_size - 2))  # length=self.kernel_size will out of index, because of sub_pixel interpolation
        # kernel = generate_kernel_trajectory(kernel_size=self.kernel_size, init_angle=init_angle, length=init_length)
        if init_length == 0:
            blurred = copy.deepcopy(sharp_patch)
        else:
            kernel_name = 'angle_%s_length_%s' % (init_angle, init_length)
            kernel = self.kernel[kernel_name]
            delta = (self.opt.fineSize - self.kernel_size) // 2
            tmp_kernel = np.pad(kernel, (delta, delta + 1), 'constant')  # pad the kernel to 256 × 256

            # patch_gamma = np.sign(sharp_patch) * (np.abs(sharp_patch)) ** self.gamma
            # patch_gamma[:, :, 0] = np.array(signal.fftconvolve(patch_gamma[:, :, 0], tmp_kernel, 'same'))
            # patch_gamma[:, :, 1] = np.array(signal.fftconvolve(patch_gamma[:, :, 1], tmp_kernel, 'same'))
            # patch_gamma[:, :, 2] = np.array(signal.fftconvolve(patch_gamma[:, :, 2], tmp_kernel, 'same'))
            # blur_patch = np.sign(patch_gamma) * (np.abs(patch_gamma)) ** (1 / self.gamma)

            patch = copy.deepcopy(sharp_patch)
            patch[:, :, 0] = np.array(signal.fftconvolve(patch[:, :, 0], tmp_kernel, 'same'))
            patch[:, :, 1] = np.array(signal.fftconvolve(patch[:, :, 1], tmp_kernel, 'same'))
            patch[:, :, 2] = np.array(signal.fftconvolve(patch[:, :, 2], tmp_kernel, 'same'))

            blurred = cv2.normalize(patch, patch, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        sharp_patch = np.transpose(sharp_patch, (2, 0, 1))
        blurred = np.transpose(blurred, (2, 0, 1))
        return {'sharp': sharp_patch, 'sharp_paths': data_path, 'blur': blurred, 'sharp_patch_VL': sharp_patch_VL}

    def __len__(self):
        return len(self.data_paths)

    def name(self):
        return 'TrainDataset'
