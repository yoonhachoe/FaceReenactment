import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import os
from typing import List, Optional, Tuple, Union
from glob import glob
import dnnlib
import numpy as np
import legacy
import mapping_net

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

    return img

def test():
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #transform = transforms.Compose([transforms.Resize(1024), transforms.ToTensor()])
    transform = transforms.Compose([transforms.Resize(580), transforms.ToTensor()])
    os.makedirs("test", exist_ok=True)
    imgdir = sorted(glob('./landmark_obama/obama1*.png'))

    for i in range(2735):
        print(imgdir[i])
        image = Image.open(imgdir[i])
        image = image.convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)

        model = mapping_net.MappingNetwork()
        model.load_state_dict(torch.load("./model/20220208-155149/model_030epochs.pth"))
        model.to(device)
        model.eval()
        with torch.no_grad():
            pred = model.forward(image)
        pred = torch.reshape(pred, (1, 16, 512))
        syn_img = generate_images(network_pkl="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl", z=pred)
        syn_img = (syn_img + 1.0) / 2.0
        save_image(syn_img.clamp(0, 1), "./test_obama1/{frame:04d}.png".format(frame=i))

if __name__ == "__main__":
    test()