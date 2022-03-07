import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from math import log10
import os
import re
from typing import List, Optional, Tuple, Union
from glob import glob
import dnnlib
import numpy as np
import legacy
import mapping_net_opt
import pickle
import lpips

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

def generate_images(network_pkl, z, translate=(0,0), rotate=0):
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    img = G.synthesis(z)

    return img

def embedding_function():

    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    transform = transforms.Compose([transforms.Resize(1024), transforms.ToTensor()])
    latents_list = []
    imgdir = sorted(glob('./frame_angry_cut/*.png'))

    for i in range(954):
        print(imgdir[i])
        image = Image.open(imgdir[i])
        image = image.convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)

        # MSE loss object
        MSE_loss = nn.MSELoss(reduction="mean")

        model = mapping_net_opt.MappingNetwork()
        model.load_state_dict(torch.load("./model_last_5000.pth"))
        model.to(device)
        model.eval()
        if i == 0 or i == 115 or i == 232 or i == 336 or i == 447 or i == 569 or i == 699 or i == 822: # angry
        # if i == 0 or i == 105 or i == 212 or i == 316 or i == 419 or i == 529 or i == 648 or i == 773: # calm
        # if i == 0 or i == 115 or i == 231 or i == 347 or i == 461 or i == 589 or i == 711 or i == 831: # disgust
        # if i == 0 or i == 109 or i == 217 or i == 319 or i == 429 or i == 578 or i == 704 or i == 822: # fearful
        # if i == 0 or i == 103 or i == 206 or i == 311 or i == 421 or i == 530 or i == 638 or i == 755: # happy
        # if i == 0 or i == 98 or i == 197 or i == 294: # neutral
        # if i == 0 or i == 114 or i == 220 or i == 324 or i == 424 or i == 537 or i == 647 or i == 758: # sad
        # if i == 0 or i == 101 or i == 198 or i == 303 or i == 400 or i == 502 or i == 608 or i == 712: #surprise
            latents = model.forward(image)
            latents_copy = latents.clone().detach()
            latents_copy = torch.reshape(latents_copy, (1, 16, 512))
            latents_copy.requires_grad = True
        #latents_rand = torch.zeros((1, 512), requires_grad=True, device=device)

        # Optimizer to change latent code in each backward step
        optimizer = optim.Adam({latents_copy}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        #optimizer = optim.Adam({latents_rand}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        # Loop to optimise latent vector to match the generated image to input image
        loss_ = []
        loss_psnr = []
        for e in range(1500):
            optimizer.zero_grad()
            syn_img = generate_images(network_pkl="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl", z=latents_copy)
            #syn_img = generate_images(network_pkl="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl", z=latents_rand)
            syn_img = (syn_img + 1.0) / 2.0
            mse = MSE_loss(syn_img, image)
            loss_fn_alex = lpips.LPIPS(net='alex')
            loss_fn_alex.to(device)
            p = loss_fn_alex(syn_img, image)
            psnr = PSNR(mse, flag=0)
            loss = mse + p
            if loss < 0.03:
                save_image(syn_img.clamp(0, 1), "./opt_angry/optimized_angry{frame:04d}_{epoch}.png".format(frame=i, epoch=e + 1))
                break
            loss.backward()
            optimizer.step()
            loss_np = loss.detach().cpu().numpy()
            loss_np = loss_np[0,0,0,0]
            print(loss_np)
            loss_psnr.append(psnr)
            loss_.append(loss_np)
            if e+1 == 1500:
                save_image(syn_img.clamp(0, 1), "./opt_angry/optimized_angry{frame:04d}_{epoch}.png".format(frame=i, epoch=e + 1))

        latents_copy_list = latents_copy.detach().clone()
        latents_list.append(latents_copy_list)

    with open('latents_angry.pkl', 'wb') as f:
        pickle.dump(latents_list, f)

if __name__ == "__main__":
    embedding_function()


