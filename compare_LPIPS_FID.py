import lpips
import torch
from torchvision import transforms
import os
from glob import glob
from PIL import Image

def calc_metrics():
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    transform = transforms.Compose([transforms.ToTensor()])

    imgdir = sorted(glob('./imgdir/*.png'))
    editdir = sorted(glob('./editdir/*.png'))
    lpips_list = []

    for i in range(len(imgdir)):
        image = Image.open(imgdir[i])
        image = image.convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)

        edit_img = Image.open(editdir[i])
        edit_img = edit_img.convert("RGB")
        edit_img = transform(edit_img)
        edit_img = edit_img.unsqueeze(0)
        edit_img = edit_img.to(device)

        lpips_alex = lpips.LPIPS(net='alex')
        lpips_alex.to(device)
        lpips_loss = lpips_alex(edit_img, image)
        lpips_loss = lpips_loss.squeeze()
        lpips_list.append(lpips_loss)
        print(lpips_loss)

    print("Nr. entries:", len(lpips_list))
    avg_lpips = sum(lpips_list)/len(lpips_list)
    print("Average LPIPS value:", avg_lpips)

    """
    For calculation of FID run:
        python -m pytorch_fid imgdir editdir --device cuda:1
        python -m pytorch_fid imgdir_ours editdir_ours --device cuda:1
    https://github.com/mseitzer/pytorch-fid
    """

if __name__ == "__main__":
    calc_metrics()