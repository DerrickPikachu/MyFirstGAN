import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ICLEVRLoader
import parameter
from parameter import device
from model import *
from evaluator import evaluation_model
import matplotlib.pyplot as plt


def train_model(generator, discriminator, g_optimizer, d_optimizer, dataloader, loss_f):
    eval_model = evaluation_model()

    for e in range(1, parameter.epochs + 1):
        print(f'Epoch {e}/{parameter.epochs}:')
        print('-' * 10)

        for img, label in tqdm(dataloader):
            img = img.to(device)
            label = label.to(device)

            ##############################################
            #                Discriminator               #
            #                    Update                  #
            ##############################################
            discriminator.train()
            d_optimizer.zero_grad()

            real_target = torch.ones(img.size(0), 1).to(device)
            fake_target = torch.zeros(img.size(0), 1).to(device)

            noise_vector = torch.randn(img.size(0), parameter.latent_size, device=device, dtype=torch.float)
            fake_img = generator((noise_vector, label))

            d_real_loss = loss_f(discriminator((img, label)), real_target)
            d_fake_loss = loss_f(discriminator((fake_img.detach(), label)), fake_target)

            d_total_loss = (d_real_loss + d_fake_loss) / 2
            d_total_loss.backward()
            d_optimizer.step()

            ##############################################
            #                  Generator                 #
            #                   Update                   #
            ##############################################
            discriminator.eval()
            generator.train()
            g_optimizer.zero_grad()

            g_loss = loss_f(discriminator((fake_img, label)), real_target)

            g_loss.backward()
            g_optimizer.step()

        ##############################################
        #               Evaluate model               #
        ##############################################
        acc, img = test_model(generator, eval_model)
        print(f'acc: {acc}')
        img = img.detach().cpu().numpy().transpose((1, 2, 0))
        plt.imshow(img)
        plt.show()


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
    g = CGenerator(parameter.latent_size).to(device)
    d = CDiscriminator().to(device)

    g_opt = torch.optim.SGD(g.parameters(), lr=parameter.lr)
    d_opt = torch.optim.SGD(d.parameters(), lr=parameter.lr)

    loss_f = torch.nn.MSELoss()

    data = ICLEVRLoader('jsonfile', mode='train')
    loader = DataLoader(data, batch_size=parameter.batch_size)

    train_model(g, d, g_opt, d_opt, loader, loss_f)

    # eval_model = evaluation_model()
    # test_model(g, eval_model)



