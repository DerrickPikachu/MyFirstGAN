import json
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from imgTransform import ImgToTorch
from parameter import image_size, batch_size, workers
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def get_iCLEVR_data(root_folder, mode):
    if mode == 'train':
        # Read the data from json file
        img_data = json.load(open(os.path.join(root_folder, 'train.json')))
        obj = json.load(open(os.path.join(root_folder, 'objects.json')))
        img = list(img_data.keys())
        label = list(img_data.values())
        # Change the string label to the one hot class vector
        # according to the object.json file
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        img_data = json.load(open(os.path.join(root_folder, f'{mode}.json')))
        obj = json.load(open(os.path.join(root_folder, 'objects.json')))
        label = img_data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder, trans, cond=False, mode='train'):
        self.root_folder = root_folder
        self.trans = trans
        self.mode = mode
        # img_list contain the filename when the mode is train,
        # otherwise it will just be None
        # label_list contain the one hot encoding vector label
        self.img_list, self.label_list = get_iCLEVR_data(root_folder, mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))

        # TODO: What is cond used for?
        self.cond = cond
        self.num_classes = 24

    def __len__(self):
        """return the size of dataset"""
        return len(self.label_list)

    def __getitem__(self, index):
        img = None

        if self.mode == 'train':
            img = Image.open('./images/' + self.img_list[index])
            img = img.convert('RGB')
            label = self.label_list[index]
        else:
            label = self.label_list[index]

        if img is not None:
            img = self.trans(img).type(torch.float)
        else:
            img = 0

        return img, torch.from_numpy(label).type(torch.float)


if __name__ == '__main__':
    dataset = ICLEVRLoader('./jsonfile/', trans=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), mode='train')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Training Images')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True), (1, 2, 0)))
    plt.show()
    print(dataset[0].size)

    # torch_img, _ = dataset[10]
    # array = torch_img.numpy()
    # import matplotlib.pyplot as plt
    # plt.imshow(array)
    # plt.show()
    # img.show()
