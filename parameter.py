import torch.cuda

device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_size = 100
lr = 0.01
epochs = 300
batch_size = 128

image_size = 64
workers = 2

gf_size = 64
df_size = 64