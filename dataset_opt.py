from torch.utils.data import Dataset
from PIL import Image
import torch
import pickle

class MyDataset(Dataset):
    def __init__(self, data_path_list, transform):
        self.path_list = data_path_list
        with open('latents_RAVDESS_final.pkl', 'rb') as f:
            self.latent = pickle.load(f)
        print(len(self.latent))
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
        latent = self.latent[idx]

        return image, latent


