import torch
import torch.nn as nn


# ------------------------------------------
class MappingNetwork(torch.nn.Module):
    def __init__(self,
                 #in_size=1024 * 1024,  # Number of input features.
                 in_size=580 * 580,  # Number of input features.
                 out_size=16 * 512,
                 ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.device = torch.device('cuda')

        self.encoder = nn.Sequential(
            # nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(32, 16, kernel_size=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            #
            # nn.Flatten(),
            # nn.Linear(1024, 768),
            # nn.BatchNorm1d(768),
            # nn.ReLU(),
            #
            # nn.Linear(768, self.out_size)

            # nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            #
            # nn.Flatten(),
            # nn.Linear(2592, 768),
            # nn.BatchNorm1d(768),
            # nn.ReLU(),
            #
            # nn.Linear(768, self.out_size)

            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(8192, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),

            nn.Linear(768, self.out_size)
        )

    def forward(self, x):
        x = x.to(self.device)
        latents = self.encoder(x)
        return latents

    def __repr__(self):
        return "Network that maps a facial keypoint image to a target StyleGAN3 latent vector w"
