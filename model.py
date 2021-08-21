import torch
from torch import nn
from dataset import ICLEVRLoader
from parameter import gf_size, df_size, device, latent_size


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


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(nz, gf_size * 8, 4),
            nn.BatchNorm2d(gf_size * 8),
            nn.ReLU(True),
        )
        # State: (gf_size * 8 x 4 x 4)
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(gf_size * 8, gf_size * 4, 4, 2, 1),
            nn.BatchNorm2d(gf_size * 4),
            nn.ReLU(True),
        )
        # State: (gf_size * 4 x 8 x 8)
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(gf_size * 4, gf_size * 2, 4, 2, 1),
            nn.BatchNorm2d(gf_size * 2),
            nn.ReLU(True),
        )
        # State: (gf_size * 2 x 16 x 16)
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(gf_size * 2, gf_size, 4, 2, 1),
            nn.BatchNorm2d(gf_size),
            nn.ReLU(True),
        )
        # State: (gf_size x 32 x 32)
        self.last = nn.Sequential(
            nn.ConvTranspose2d(gf_size, 3, 4, 2, 1),
            nn.Tanh()
        )
        # State: (3 x 64 x 64)

    def forward(self, input):
        latent, label = input
        latent = torch.cat([latent, label.view(-1, 24, 1, 1)], dim=1)
        out = self.l1(latent)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.last(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()

        self.label_layer = nn.Linear(in_features=24, out_features=64 * 64)

        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, df_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # State: (df_size x 32 x 32)
        self.l2 = nn.Sequential(
            nn.Conv2d(df_size, df_size * 2, 4, 2, 1),
            nn.BatchNorm2d(df_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # State: (df_size * 2 x 16 x 16)
        self.l3 = nn.Sequential(
            nn.Conv2d(df_size * 2, df_size * 4, 4, 2, 1),
            nn.BatchNorm2d(df_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # State: (df_size * 4 x 8 x 8)
        self.l4 = nn.Sequential(
            nn.Conv2d(df_size * 4, df_size * 8, 4, 2, 1),
            nn.BatchNorm2d(df_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # State: (df_size * 8 x 4 x 4)
        self.last = nn.Sequential(
            nn.Conv2d(df_size * 8, 1, 4),
            nn.Sigmoid(),
        )
        # State: (1 x 1 x 1)

    def forward(self, input):
        img, label = input
        label = self.label_layer(label).view(-1, 1, 64, 64)
        img = torch.cat([img, label], dim=1)
        out = self.l1(img)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.last(out)
        return out


class SAGenerator(nn.Module):
    def __init__(self, nz):
        super(SAGenerator, self).__init__()

        self.dc_generator = Generator(nz + 24)
        self.attn1 = Self_Attn(gf_size * 2, 'relu')
        self.attn2 = Self_Attn(gf_size, 'relu')

    def forward(self, input):
        latent_code, label = input
        # Conditional
        latent_code = torch.cat([latent_code, label.view(-1, 24, 1, 1)], dim=1)
        out = self.dc_generator.l1(latent_code)
        out = self.dc_generator.l2(out)
        out = self.dc_generator.l3(out)
        out, _ = self.attn1(out)
        out = self.dc_generator.l4(out)
        out, _ = self.attn2(out)
        out = self.dc_generator.last(out)
        return out


class SADiscriminator(nn.Module):
    def __init__(self, in_dim):
        super(SADiscriminator, self).__init__()

        self.label_layer = nn.Linear(in_features=24, out_features=64 * 64)
        self.dc_discriminator = Discriminator(in_dim)
        self.attn1 = Self_Attn(df_size * 4, 'relu')
        self.attn2 = Self_Attn(df_size * 8, 'relu')

    def forward(self, input):
        img, label = input
        # Conditional
        label = self.label_layer(label).view(-1, 1, 64, 64)
        img = torch.cat([img, label], dim=1)
        out = self.dc_discriminator.l1(img)
        out = self.dc_discriminator.l2(out)
        out = self.dc_discriminator.l3(out)
        out, _ = self.attn1(out)
        out = self.dc_discriminator.l4(out)
        out, _ = self.attn2(out)
        out = self.dc_discriminator.last(out)
        return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


if __name__ == "__main__":
    g = SAGenerator(latent_size)
    d = SADiscriminator()
    print(g)
    print(d)
