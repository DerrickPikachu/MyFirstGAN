import torch.cuda

device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_size = 100
lr = 0.01
epochs = 300
batch_size = 16
