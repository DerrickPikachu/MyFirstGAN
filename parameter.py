import torch.cuda
from torch import nn, optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_size = 64
lr = 0.0002
epochs = 60
batch_size = 128

image_size = 64
workers = 2

gf_size = 64
df_size = 64

beta1 = 0.5

loss_f = nn.BCELoss()
fixed_noise = torch.randn(64, latent_size, 1, 1, device=device)
