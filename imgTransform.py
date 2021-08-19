import numpy as np
import torch


class ImgToTorch(object):
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray):
        new_img = img.transpose((2, 0, 1)).astype('float')
        new_img /= 255
        return torch.from_numpy(new_img)
