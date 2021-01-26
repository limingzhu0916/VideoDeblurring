from .base_dataset import BaseDataset
import numpy as np
from util.util import make_dataset
import cv2
from dataset.generate_kernel import generate_kernel_trajectory
from scipy import signal
import albumentations as albu

class TrainDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_paths = sorted(make_dataset(opt.data_root))
        self.gamma = 2.2
        self.kernel_size = opt.kernel_size

    def __getitem__(self, index):
        """
        import the images from the data root, then crop them into 256 × 256, random
        blur kernel from kernel set, and conv blur kernel with patch
        """
        data_path = self.data_paths[index]
        image = cv2.imread(data_path)
        patch = albu.RandomCrop(self.opt.fineSize, self.opt.fineSize, always_apply=True)(image=image)['image']
        # cv2.imwrite('./path.png', patch)
        patch = cv2.normalize(patch, patch, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        init_angle = np.random.uniform(0, 180)
        init_length = np.random.uniform(0, self.kernel_size - 2)  # length=self.kernel_size will out of index, because of sub_pixel interpolation
        kernel = generate_kernel_trajectory(kernel_size=self.kernel_size, init_angle=init_angle, length=init_length)
        delta = (self.opt.fineSize - self.kernel_size) // 2
        tmp_kernel = np.pad(kernel, (delta, delta + 1), 'constant')  # pad the kernel to 256 × 256

        patch_gamma = np.sign(patch) * (np.abs(patch)) ** self.gamma
        patch_gamma[:, :, 0] = np.array(signal.fftconvolve(patch_gamma[:, :, 0], tmp_kernel, 'same'))
        patch_gamma[:, :, 1] = np.array(signal.fftconvolve(patch_gamma[:, :, 1], tmp_kernel, 'same'))
        patch_gamma[:, :, 2] = np.array(signal.fftconvolve(patch_gamma[:, :, 2], tmp_kernel, 'same'))
        patch = np.sign(patch_gamma) * (np.abs(patch_gamma)) ** (1 / self.gamma)

        blurred = cv2.normalize(patch, patch, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.imwrite('./blur_patch.png', np.abs(blurred) * 255)

        return {'sharp': patch, 'sharp_paths': data_path, 'blur': blurred}

    def __len__(self):
        return len(self.data_paths)

    def name(self):
        return 'TrainDataset'
