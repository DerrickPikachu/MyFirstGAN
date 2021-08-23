import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from parameter import *

from dataset import ICLEVRLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

'''===============================================================
1. Title:     

DLP spring 2021 Lab7 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled 
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]

==============================================================='''


class evaluation_model:
    def __init__(self):
        # modify the path to your own path
        checkpoint = torch.load('classifier_weight.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24

    def compute_acc(self, out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
        return acc / total

    def eval(self, images, labels):
        with torch.no_grad():
            # your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc


def test_model(generator, eval_model, epoch):
    generator.eval()

    test_data = ICLEVRLoader('jsonfile', trans=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), mode='test')
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    acc = 0

    for _, label in test_loader:
        label = label.to(device)
        latent = torch.randn(label.size(0), nz, 1, 1, device=device, dtype=torch.float)
        generated_img = generator((latent, label))
        acc = eval_model.eval(generated_img, label)

    # plt.imshow(np.transpose(vutils.make_grid(generated_img.cpu(), padding=2, normalize=True), (1, 2, 0)))
    # plt.savefig(f'record/record{epoch}.jpg')
    # plt.show()

    return acc, generated_img.detach().cpu()


if __name__ == "__main__":
    gen = torch.load('generator81.pth')
    eval_model = evaluation_model()
    best_acc = 0
    best_result = None
    for _ in range(1000):
        acc, gen_img = test_model(gen, eval_model, 0)
        if acc > best_acc:
            best_acc = acc
            best_result = gen_img
    print(f'acc: {best_acc}')
    plt.imshow(np.transpose(vutils.make_grid(gen_img.cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(f'record/best_result.jpg')
