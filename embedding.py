import click
import os
import numpy as np
import datetime
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from glob import glob
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import dnnlib
import legacy
import mapping_net_opt
from dataset import MyDataset

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m
# ----------------------------------------------------------------------------
def generate_images(network_pkl, z, translate=(0,0), rotate=0):
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    img = G.synthesis(z)
    #img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    return img
# ----------------------------------------------------------------------------
@click.command()
@click.option('--epochs', help='Number of epochs to train', type=int, required=True)
@click.option('--lr', help='Learning rate', type=float, default=3e-4)
@click.option('--betas', help='Momentum parameters for Adam', type=tuple, default=(0.9, 0.999))
@click.option('--batch_size', help='Batch size for train- and valloader', type=int, default=8)
@click.option('--reg', help='L2 Regularization strength as Adam weight decay', type=float, default=1e-4)
def train(
        epochs: int,
        lr: float,
        betas: tuple,
        batch_size: int,
        reg: float
):
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    writer = SummaryWriter(log_dir="tensorboard_logs/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    transform = transforms.Compose([transforms.Resize(1024), transforms.ToTensor()])
    os.makedirs("opt", exist_ok=True)


    DATA_PATH_LIST = sorted(glob('./rgb_5000/*.png'))
    dataset = MyDataset(DATA_PATH_LIST, transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # define model
    model = mapping_net_opt.MappingNetwork()
    model.to(device)

    # define loss
    MSE_loss = nn.MSELoss(reduction="mean")

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr, betas, weight_decay=reg)

    for e in range(epochs):
        sum_train_loss = 0.0
        sum_val_loss = 0.0
        print("Epoch:", e)
        for i, item in enumerate(trainloader):
            image, latent, seed = item
            #image, latent = item
            image = image.to(device)
            latent = latent.squeeze(1).flatten(1)
            latent = latent.to(device) #batch * 8192
            seed = seed.to(device)
            optimizer.zero_grad()
            pred = model.forward(image) #batch * 8192

            train_loss = MSE_loss(pred, latent)
            train_loss.backward()
            optimizer.step()
            sum_train_loss += train_loss.item()

        writer.add_scalar('L2 train loss', sum_train_loss / len(trainloader), e)
        print('train loss', sum_train_loss / len(trainloader))

        for i, item in enumerate(testloader):
            image, latent, seed = item
            #image, latent = item
            image = image.to(device)
            latent = latent.squeeze(1).flatten(1)
            latent = latent.to(device) #batch * 8192
            seed = seed.to(device)
            pred = model.forward(image) #batch * 8192

            val_loss = MSE_loss(pred, latent)
            sum_val_loss += val_loss.item()

        writer.add_scalar('L2 val loss', sum_val_loss / len(testloader), e)
        print('val loss', sum_val_loss / len(testloader))
    torch.save(model.state_dict(), './model_last_5000.pth')

# ---------------------------------

if __name__ == "__main__":
    train()

