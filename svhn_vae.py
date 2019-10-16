import torch
from torch import nn, optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

"""Check if a CUDA GPU is available, and if yes use it. Else use the CPU for computations."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using %s for computation" % device)

project_dir = "VAE/"
dataset_dir = project_dir + "datasets/"
images_dir = project_dir + "images/"
model_dir = project_dir + "model/"

batch_size = 32  # number of inputs in each batch
epochs = 10  # times to run the model on complete data
image_size = 32  # dimension of the image
hidden_size = 1024  # hidden dimension
latent_size = 32  # latent vector dimension
lr = 1e-3  # learning rate
train_loss = []

train_data = datasets.SVHN(
    dataset_dir + "SVHN/",
    split="train",
    download=True,
    transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor()]),
)
test_data = datasets.SVHN(
    dataset_dir + "SVHN/",
    split="test",
    download=True,
    transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor()]),
)

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)


def show_images(images):
    images = torchvision.utils.make_grid(images)
    show_image(images)


def show_image(img):
    plt.imshow(transforms.ToPILImage()(img), interpolation="bicubic")
    plt.show()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 1024, 1, 1)


class VAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        image_dim=image_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
    ):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2),
            nn.LeakyReLU(0.2),
            Flatten(),
        )
        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(hidden_size, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 6, 2),
            nn.Sigmoid(),
        )

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.encoder(x)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        z = self.sample(log_var, mean)
        x = self.fc(z)
        x = self.decoder(x)

        return x, mean, log_var


vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=lr)

"""
Unomment this out if you dont want to train from scratch. Ensure that the VAE.pt file is stored in the models sub-directory
"""
# vae.load_state_dict(torch.load(model_dir+"SVHNVAE.pt"))

"""
Set the model to the training mode first and train
"""
vae.train()

for epoch in range(epochs):
    for i, (images, _) in enumerate(trainloader):
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed_image, mean, log_var = vae(images)
        CE = F.binary_cross_entropy(reconstructed_image, images, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = CE + KLD
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()

        if i % 100 == 0:
            print("Loss:")
            print(loss.item() / len(images))

plt.plot(train_loss)
plt.show()

"""
Set the model to the evaluation mode. This is important otherwise you will get inconsistent results. Then load data from the test set.
"""
vae.eval()
vectors = []
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        reconstructed_image, mean, log_var = vae(images)
        temp = list(zip(labels.tolist(), mean.tolist()))
        for x in temp:
            vectors.append(x)
        if i % 100 == 0:
            show_images(reconstructed_image.cpu())
            img_name = images_dir + "evaluation/" + str(i).zfill(3)
            plt.savefig(img_name)
            plt.show()

"""
Using Singular Value Decomposition, visualise the two largest eigenvalues. Add the labels for each element and create a dataframe.
"""
labels, z_vectors = list(zip(*vectors))
z_vectors = torch.tensor(z_vectors)
# z_mean = torch.mean(torch.tensor(z_vectors), 0)
# z_vectors.sub_(z_mean.expand_as(z_vectors))
U, S, V = torch.svd(torch.t(z_vectors))
C = torch.mm(z_vectors, U[:, :2]).tolist()
C = [x + [labels[i]] for i, x in enumerate(C)]

df = pd.DataFrame(C, columns=["x", "y", "label"])
df.head()

sns.lmplot(x="x", y="y", data=df, fit_reg=False, hue="label")

"""
Save the model incase we need to load it again.
"""
torch.save(vae.state_dict(), model_dir + "VAE.pt")

