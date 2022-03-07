import torch
from torchvision.utils import save_image
from math import log10
import os
import re
from typing import List, Optional, Tuple, Union
import dnnlib
import numpy as np
import legacy
import pickle

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

def visualize_images():
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    #Choose latents to visualize
    with open('latents_edited_happy.pkl', 'rb') as f:
        latent = pickle.load(f)
    #Choose boundary
    with open('boundary.pkl', 'rb') as g:
        boundary = pickle.load(g)
    boundary = torch.from_numpy(boundary).to(device)

    for i in range(len(latent)):
        input = latent[i].reshape(1, -1).to(device)
        syn_img = generate_images(
            network_pkl="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl",
            z=input)
        syn_img = (syn_img + 1.0) / 2.0
        save_image(syn_img.clamp(0, 1), "./edited/happy_{latent:02d}.png".format(latent=i))

if __name__ == "__main__":
    visualize_images()

