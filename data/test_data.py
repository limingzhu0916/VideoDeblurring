from data.base_dataset import BaseDataset
from util.util import make_dataset
import cv2

class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.data_paths = sorted(make_dataset(opt.data_root))
        self.sharp_paths = sorted(make_dataset(opt.sharp_root))

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        sharp_path = self.sharp_paths[index]
        image = cv2.imread(data_path)
        sharp_image = cv2.imread(sharp_path)
        blurred = cv2.normalize(image, image, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return {'blurred': blurred, 'blur_path': data_path, 'sharp': sharp_image}

    def __len__(self):
        return len(self.data_paths)

    def name(self):
        return 'TestDataset'



