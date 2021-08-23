import copy
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ICLEVRLoader
import parameter
from parameter import *
from model import *
from evaluator import evaluation_model
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from evaluator import test_model


def rand_fake_c(b_size):
    fake_c = torch.zeros((b_size, 24))
    for b in range(b_size):
        fake_c[b][random.randint(0, 23)] = 1
        for i in range(2):
            if random.random() > 0.5:
                fake_c[b][random.randint(0, 23)] = 1
    return fake_c.to(device)


if __name__ == "__main__":
    data = ICLEVRLoader('jsonfile', trans=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), mode='train')
    loader = DataLoader(data, batch_size=batch_size, num_workers=workers)

    # Build network
    eval_model = evaluation_model()
    # netG = Generator(ngpu).to(device)
    netG = SAGenerator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)
    setup(netG)
    setup(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr * 5, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    acc_list = []
    iters = 0

    best_acc = 0
    best_weight = None

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        accumulate_acc = 0
        cal_acc_counter = 0
        for i, data in enumerate(loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            c_label = data[1].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD((real_cpu, c_label)).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG((noise, c_label))
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD((fake.detach(), c_label)).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            output = netD((real_cpu, rand_fake_c(b_size))).view(-1)
            errD_fake_c = criterion(output, label)
            errD_fake_c.backward()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake + errD_fake_c
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            D_G_z2 = 0
            for _ in range(1):
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost

                # noise = torch.randn(b_size, nz, 1, 1, device=device)
                # fake = netG((noise, c_label))

                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD((fake, c_label)).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 += output.mean().item()
                # Update G
                optimizerG.step()
            D_G_z2 /= 3

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 50 == 0) or ((epoch == num_epochs - 1) and (i == len(loader) - 1)):
                with torch.no_grad():
                    # fake = netG(fixed_noise).detach().cpu()
                    acc, gen_img = test_model(netG, eval_model, epoch)
                # plt.imshow(np.transpose(vutils.make_grid(gen_img, padding=2, normalize=True), (1, 2, 0)))
                # plt.show()
                # plt.savefig(f'record/record{iters}')
                print(f'acc: {acc}')

                accumulate_acc += acc
                cal_acc_counter += 1

            if acc > best_acc:
                best_acc = acc
                best_weight = copy.deepcopy(netG.state_dict())

            iters += 1

        acc_list.append(accumulate_acc / cal_acc_counter)

    print(f'best acc: {best_acc}')
    # netG.load_state_dict(best_weight)
    # torch.save(netG, 'generator3.pth')

    # plt.figure()
    plt.title('SAGAN testing accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(acc_list)
    # plt.show()
    plt.savefig('record/curve.jpg')

