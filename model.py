import torch
from torch import nn
from dataset import ICLEVRLoader
from parameter import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def setup(net):
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    net.apply(weights_init)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + 24, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        latent, label = input
        latent = torch.cat((latent, label.view(-1, 24, 1, 1)), dim=1)
        return self.main(latent)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.l_y = nn.Sequential(
            nn.Linear(24, ndf * 2 * 16 * 16)
        )

        self.first_conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.second_conv = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2 * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Flatten(),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=ndf * 8 * 4 * 4, out_features=1024),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=1024, out_features=512),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=512, out_features=128),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=128, out_features=1),
        #     nn.Sigmoid()
        # )

        # self.main = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        img, label = input
        label = self.l_y(label)
        x = self.first_conv(img)
        x = torch.cat((x, label.view(-1, ndf * 2, 16, 16)), dim=1)
        out = self.second_conv(x)
        # out = self.classifier(out)
        return out


# class SAGenerator(nn.Module):
#     def __init__(self, nz):
#         super(SAGenerator, self).__init__()
#
#         self.dc_generator = Generator(nz)
#         self.attn1 = Self_Attn(gf_size * 2, 'relu')
#         self.attn2 = Self_Attn(gf_size, 'relu')
#
#     def forward(self, input):
#         latent_code, label = input
#         # Conditional
#         latent_code = torch.cat([latent_code, label.view(-1, 24, 1, 1)], dim=1)
#         out = self.dc_generator.l1(latent_code)
#         out = self.dc_generator.l2(out)
#         out = self.dc_generator.l3(out)
#         out, _ = self.attn1(out)
#         out = self.dc_generator.l4(out)
#         out, _ = self.attn2(out)
#         out = self.dc_generator.last(out)
#         return out
#
#
# class SADiscriminator(nn.Module):
#     def __init__(self, in_dim):
#         super(SADiscriminator, self).__init__()
#
#         self.label_layer = nn.Linear(in_features=24, out_features=64 * 64)
#         self.dc_discriminator = Discriminator(in_dim)
#         self.attn1 = Self_Attn(df_size * 4, 'relu')
#         self.attn2 = Self_Attn(df_size * 8, 'relu')
#
#     def forward(self, input):
#         img, label = input
#         # Conditional
#         label = self.label_layer(label).view(-1, 1, 64, 64)
#         img = torch.cat([img, label], dim=1)
#         out = self.dc_discriminator.l1(img)
#         out = self.dc_discriminator.l2(out)
#         out = self.dc_discriminator.l3(out)
#         out, _ = self.attn1(out)
#         out = self.dc_discriminator.l4(out)
#         out, _ = self.attn2(out)
#         out = self.dc_discriminator.last(out)
#         return out
#
#
# class Self_Attn(nn.Module):
#     """ Self attention Layer"""
#
#     def __init__(self, in_dim, activation):
#         super(Self_Attn, self).__init__()
#         self.chanel_in = in_dim
#         self.activation = activation
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)  #
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         m_batchsize, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
#         energy = torch.bmm(proj_query, proj_key)  # transpose check
#         attention = self.softmax(energy)  # BX (N) X (N)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#
#         out = self.gamma * out + x
#         return out, attention


if __name__ == "__main__":
    # g = SAGenerator(latent_size)
    # d = SADiscriminator()
    # print(g)
    # print(d)
    pass
