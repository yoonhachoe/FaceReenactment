from torch.utils.data import Dataset
from PIL import Image
import torch
import get_latents

class MyDataset(Dataset):
    def __init__(self, data_path_list, transform):
        self.path_list = data_path_list
        self.network = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"
        self.latent = get_latents.get_latents(self.network, seeds=list(range(5000)), outdir="out")
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.path_list[idx])
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        #latent = self.latent[idx]
        latent = self.latent[idx][0]
        seed = self.latent[idx][1]
        return image, latent, seed
        #return image, latent

