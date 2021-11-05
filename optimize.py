import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision
from torchvision import models
from torchvision.utils import save_image
import numpy as np
from math import log10
import matplotlib.pyplot as plt
import os
import re
from typing import List, Optional, Tuple, Union
import click
import dnnlib
import numpy as np
import PIL.Image
import legacy
import pickle



class VGG16_perceptual(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16_perceptual, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu1_2 = h
        h = self.slice3(h)
        h_relu3_2 = h
        h = self.slice4(h)
        h_relu4_2 = h
        return h_relu1_1, h_relu1_2, h_relu3_2, h_relu4_2

def loss_function(syn_img, img, img_p, MSE_loss, upsample, perceptual):
  #UpSample synthesized image to match the input size of VGG-16 input.
  #Extract mid level features for real and synthesized image and find the MSE loss between them for perceptual loss.
  #Find MSE loss between the real and synthesized images of actual size
  syn_img_p = upsample(syn_img)
  syn0, syn1, syn2, syn3 = perceptual(syn_img_p)
  r0, r1, r2, r3 = perceptual(img_p)
  mse = MSE_loss(syn_img,img)

  per_loss = 0
  per_loss += MSE_loss(syn0,r0)
  per_loss += MSE_loss(syn1,r1)
  per_loss += MSE_loss(syn2,r2)
  per_loss += MSE_loss(syn3,r3)

  return mse, per_loss

def PSNR(mse, flag = 0):
    #flag = 0 if a single image is used and 1 if loss for a batch of images is to be calculated
    if flag == 0:
        psnr = 10 * log10(1 / mse.item())
    return psnr

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

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

#----------------------------------------------------------------------------

def generate_images(network_pkl, truncation_psi, noise_mode, outdir, translate, rotate, z):
    #print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    #if G.c_dim != 0:
    #    if class_idx is None:
    #        raise click.ClickException('Must specify class label with --class when using a conditional network')
    #    label[:, class_idx] = 1
    #else:
    #    if class_idx is not None:
    #        print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    #z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    img = G.synthesis(z)
    #img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    #print(img.shape)
    return img

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--imgdir', help='Where original image locates', type=str, required=True, metavar='DIR')
def embedding_function(
    network_pkl: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    imgdir: str
):
    device = torch.device('cuda')
    with open(imgdir, "rb") as f:
        image = Image.open(f)
        image = image.convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    #print(image.shape)
    upsample = torch.nn.Upsample(scale_factor = 256/1024, mode = 'bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)
    # Perceptual loss initialise object
    perceptual = VGG16_perceptual().to(device)

    # MSE loss object
    MSE_loss = nn.MSELoss(reduction="mean")
    # since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
    latents = torch.zeros((1, 16, 512), requires_grad=True, device=device)
    # Optimizer to change latent code in each backward step
    optimizer = optim.Adam({latents}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    # Loop to optimise latent vector to match the generated image to input image
    loss_ = []
    loss_psnr = []
    for e in range(1500):
        optimizer.zero_grad()
        syn_img = generate_images(network_pkl, truncation_psi, noise_mode, outdir, translate, rotate, latents)
        syn_img = (syn_img + 1.0) / 2.0
        mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
        psnr = PSNR(mse, flag=0)
        loss = per_loss + mse
        loss.backward()
        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        loss_p = per_loss.detach().cpu().numpy()
        loss_m = mse.detach().cpu().numpy()
        loss_psnr.append(psnr)
        loss_.append(loss_np)
        if (e + 1) % 500 == 0:
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e + 1, loss_np, loss_m,
                                                                                            loss_p, psnr))
            save_image(syn_img.clamp(0, 1), "output{}.png".format(e + 1))
            # np.save("loss_list.npy",loss_)
            # np.save("latent_W.npy".format(),latents.detach().cpu().numpy())
    with open('loss.pkl', 'wb') as f:
        pickle.dump(loss_, f)

if __name__ == "__main__":
    embedding_function()

