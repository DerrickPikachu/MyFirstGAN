import torch
from torch import nn
from dataset import ICLEVRLoader
from parameter import gf_size, df_size, device


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def setup_model(net: nn.Module):
    net = net.to(device)
    net.apply(weights_init)


class CGenerator(nn.Module):
    def __init__(self, latent_size):
        super(CGenerator, self).__init__()

        self.condition_layer = nn.Sequential(
            nn.Linear(24, 16),
        )

        # TODO: survey inplace
        self.latent_layer = nn.Sequential(
            nn.Linear(latent_size, 4 * 4 * 512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.generate_layer = nn.Sequential(
            nn.ConvTranspose2d(513, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64 * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 1, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 1, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        latent, condition = input

        latent = self.latent_layer(latent).view(-1, 512, 4, 4)
        condition = self.condition_layer(condition).view(-1, 1, 4, 4)

        feature_map = torch.cat([latent, condition], dim=1)
        img = self.generate_layer(feature_map)
        return img


class CDiscriminator(nn.Module):
    def __init__(self):
        super(CDiscriminator, self).__init__()

        self.condition_layer = nn.Sequential(
            nn.Linear(24, 3 * 64 * 64),
        )

        self.discriminator_layer = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 3, 2, bias=False),
            nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 3, 2, bias=False),
            nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        img, condition = input
        condition = self.condition_layer(condition).view(-1, 3, 64, 64)
        cond_img = torch.cat([img, condition], dim=1)
        output = self.discriminator_layer(cond_img)
        return output


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, gf_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gf_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(gf_size * 8, gf_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(gf_size * 4, gf_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(gf_size * 2, gf_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(gf_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, df_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df_size, df_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df_size * 2, df_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df_size * 4, df_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":
    data = ICLEVRLoader('jsonfile')
    # img, label = data[10]

    # latent = torch.randn(100, dtype=torch.float)
    # generator = CGenerator(100)
    # generated_img = generator((latent, label)).view(3, 64, 64)
    # import matplotlib.pyplot as plt
    # generated_img = generated_img.detach().numpy().transpose((1, 2, 0))
    # plt.imshow(generated_img)
    # plt.show()
    # print(img)

    # img = img.view(-1, 3, 64, 64)
    # discriminator = CDiscriminator()
    # output = discriminator((img, label))
    # print(output)
