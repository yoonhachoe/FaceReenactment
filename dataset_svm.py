from torch.utils.data import Dataset
import torch
import pickle

class MyDatasetSVM(Dataset):
    def __init__(self, mode):
        self.mode = mode
        # Load latents for the corresponding emotions that should be edited
        if self.mode == 'happy':
            with open('latents_happy.pkl', 'rb') as f:
                self.latents_pos = pickle.load(f)
            with open('latents_sad.pkl', 'rb') as g:
                self.latents_neg = pickle.load(g)

        elif self.mode == 'sad':
            with open('latents_sad.pkl', 'rb') as f:
                self.latents_pos = pickle.load(f)
            with open('latents_happy.pkl', 'rb') as g:
                self.latents_neg = pickle.load(g)

        elif self.mode in ['calm', 'angry', 'fearful', 'disgust', 'surprise']:
            with open('latents_' + self.mode + '.pkl', 'rb') as f:
                self.latents_pos = pickle.load(f)
            with open('latents_neutral.pkl', 'rb') as g:
                self.latents_neg = pickle.load(g)

        # This part includes some manual pre-selection of frames to perform classification on
        hindex1 = list(range(12, 18))
        hindex2 = list(range(24, 39))
        hindex3 = list(range(48, 170))
        hindex4 = list(range(173, 259))
        hindex5 = list(range(262, 311))
        hindex6 = list(range(335, 422))
        hindex7 = list(range(434, 756))
        hindex8 = list(range(777, 872))
        happy_indices = [*hindex1, *hindex2, *hindex3, *hindex4, *hindex5, *hindex6, *hindex7, *hindex8]

        sindex1 = list(range(0, 71))
        sindex2 = list(range(85, 158))
        sindex3 = list(range(162, 172))
        sindex4 = list(range(182, 263))
        sindex5 = list(range(267, 269))
        sindex6 = list(range(273, 282))
        sindex7 = list(range(286, 309))
        sindex8 = list(range(324, 388))
        sindex9 = list(range(395, 758))
        sindex10 = list(range(781, 868))
        sad_indices = [*sindex1, *sindex2, *sindex3, *sindex4, *sindex5, *sindex6, *sindex7, *sindex8, *sindex9, *sindex10]

        self.latents_pos = [self.latents_pos[i] for i in happy_indices]
        self.latents_pos = torch.cat(self.latents_pos[:])
        self.latents_neg = [self.latents_neg[i] for i in sad_indices]
        self.latents_neg = torch.cat(self.latents_neg[:])
        self.latents = torch.cat((self.latents_pos, self.latents_neg))
        self.labels = torch.zeros(self.latents.size(dim=0), 1)
        self.labels[:self.latents_pos.size(dim=0)-1] = 1

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        latent = torch.flatten(self.latents[idx])
        label = self.labels[idx]

        return latent, label