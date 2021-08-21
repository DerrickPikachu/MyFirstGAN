import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ICLEVRLoader
import parameter
from parameter import device, loss_f, image_size
from model import *
from evaluator import evaluation_model
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def train_model(generator, discriminator, g_optimizer, d_optimizer, dataloader):
    eval_model = evaluation_model()

    for e in range(1, parameter.epochs + 1):
        print(f'Epoch {e}/{parameter.epochs}:')
        print('-' * 10)
        iter = 0

        for img, label in tqdm(dataloader):
            img = img.to(device)
            label = label.to(device)
            b_size = img.size(0)

            ##############################################
            #                Discriminator               #
            #                    Update                  #
            ##############################################
            # Setup discriminator
            discriminator.zero_grad()

            # Train with real image data
            # Setup target
            real_target = torch.full((b_size,), 1, dtype=torch.float, device=device)
            # discriminator predict and calculate loss
            output = discriminator(img).view(-1)
            real_loss = loss_f(output, real_target)
            # Calculate gradient
            real_loss.backward()

            # Train with fake image data
            # Generate fake img
            noise = torch.randn(b_size, parameter.latent_size, 1, 1, device=device)
            fake_img = generator(noise)
            fake_target = torch.full((b_size,), 0, dtype=torch.float, device=device)
            # Calculate fake img loss
            output = discriminator(fake_img.detach()).view(-1)
            fake_loss = loss_f(output, fake_target)
            # Calculate gradient
            fake_loss.backward()

            # Update
            d_optimizer.step()

            ##############################################
            #                  Generator                 #
            #                   Update                   #
            ##############################################
            # Setup generator and target
            generator.zero_grad()
            g_target = torch.full((b_size,), 1, dtype=torch.float, device=device)
            # Calculate loss
            output = discriminator(fake_img).view(-1)
            g_loss = loss_f(output, g_target)
            # Update generator
            g_loss.backward()
            g_optimizer.step()

            ##############################################
            #               Evaluate model               #
            ##############################################
            if iter % 50 == 0:
                with torch.no_grad():
                    fake = generator(parameter.fixed_noise).detach().cpu()
                plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
                plt.show()

            iter += 1


def test_model(generator, eval_model):
    generator.eval()

    test_data = ICLEVRLoader('jsonfile', mode='test')
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    acc = 0

    for _, label in test_loader:
        label = label.to(device)
        latent = torch.randn(label.size(0), parameter.latent_size, device=device, dtype=torch.float)
        generated_img = generator((latent, label))
        acc = eval_model.eval(generated_img, label)

    return acc, generated_img[0].view(3, 64, 64)


if __name__ == "__main__":
    data = ICLEVRLoader('jsonfile', trans=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), mode='train')
    loader = DataLoader(data, batch_size=parameter.batch_size)

    g = Generator(parameter.latent_size)
    d = Discriminator()
    setup_model(g)
    setup_model(d)

    g_opti = torch.optim.Adam(g.parameters(), lr=parameter.lr, betas=(parameter.beta1, 0.999))
    d_opti = torch.optim.Adam(d.parameters(), lr=parameter.lr, betas=(parameter.beta1, 0.999))

    train_model(g, d, g_opti, d_opti, loader)

    # eval_model = evaluation_model()
    # test_model(g, eval_model)


