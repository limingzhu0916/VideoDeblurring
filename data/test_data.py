from data.base_dataset import BaseDataset
from util.util import make_dataset
import cv2
import numpy as np

class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.data_paths = sorted(make_dataset(opt.video_root))
        self.sharp_paths = sorted(make_dataset(opt.sharp_root))
        if opt.dataset_mode == 'train' and opt.use_validation:
            self.data_paths = self.data_paths[0:20]
            self.sharp_paths = self.sharp_paths[0:20]

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        sharp_path = self.sharp_paths[index]
        image = cv2.imread(data_path)
        sharp_image = cv2.imread(sharp_path)

        blurred = cv2.normalize(image, image, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        sharp_image = cv2.normalize(sharp_image, sharp_image, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        blurred = np.pad(blurred, ((24, 24), (0, 0), (0, 0)), 'constant')
        sharp_image = np.transpose(sharp_image, (2, 0, 1))
        blurred = np.transpose(blurred, (2, 0, 1))

        return {'blurred': blurred, 'blur_path': data_path, 'sharp': sharp_image}

    def __len__(self):
        return len(self.data_paths)

    def name(self):
        return 'TestDataset'



