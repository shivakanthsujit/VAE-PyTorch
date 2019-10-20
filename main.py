import torch
from torch import nn, optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import itertools

import VAE
import DCVAE

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conv",
    help="Use the DCVAE model. Default is the Vanilla VAE",
    action="store_true",
)
arg = parser.parse_args()
use_conv = arg.conv

"""Check if a CUDA GPU is available, and if yes use it. Else use the CPU for computations."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using %s for computation" % device)

project_dir = "VAE/"
dataset_dir = project_dir + "datasets/"
images_dir = project_dir + "images/"
model_dir = project_dir + "model/"

batch_size = 32  # number of inputs in each batch
epochs = 10  # times to run the model on complete data
lr = 1e-3  # learning rate
train_loss = []

global image_size
global input_size
global hidden_size
global latent_size
global train_data
global test_data

if use_conv:
    image_size = 32  # dimension of the image
    hidden_size = 1024  # hidden dimension
    latent_size = 32  # latent vector dimension
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
else:
    image_size = 28
    input_size = image_size ** 2  # size of each input
    hidden_size = 300  # hidden dimension
    latent_size = 45  # latent vector dimension
    train_data = datasets.FashionMNIST(
        dataset_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = datasets.FashionMNIST(
        dataset_dir, train=False, download=True, transform=transforms.ToTensor()
    )

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)


def show_images(images):
    images = torchvision.utils.make_grid(images)
    show_image(images)


def show_image(img):
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.show()


if use_conv:
    vae = DCVAE.DCVAE().to(device)
else:
    vae = VAE.VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=lr)

"""
Unomment this out if you dont want to train from scratch. Ensure that the appropriate file is stored in the models sub-directory
"""
# if use_conv:
#     vae.load_state_dict(torch.load(model_dir+"DCVAE.pt"))
# else :
#     vae.load_state_dict(torch.load(model_dir+"VAE.pt"))
"""
Set the model to the training mode first and train
"""
vae.train()

for epoch in range(epochs):
    for i, (images, _) in enumerate(trainloader):
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed_image, mean, log_var = vae(images)
        if use_conv:
            CE = F.binary_cross_entropy(reconstructed_image, images, reduction="sum")
        else:
            CE = F.binary_cross_entropy(
                reconstructed_image, images.view(-1, input_size), reduction="sum"
            )
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
        reconstructed_image = reconstructed_image.view(-1, 1, image_size, image_size)

        for x in temp:
            vectors.append(x)
        if i % 100 == 0:
            show_images(reconstructed_image.cpu())

            if use_conv:
                img_name = images_dir + "evaluation/DCVAE/" + str(i).zfill(3)
            else:
                img_name = images_dir + "evaluation/VAE/" + str(i).zfill(3)

            torchvision.utils.save_image(reconstructed_image.cpu(), img_name)

if use_conv:
    """
    Using Singular Value Decomposition, visualise the two largest eigenvalues. Add the labels for each element and create a dataframe.
    """
    labels, z_vectors = list(zip(*vectors))
    z_vectors = torch.tensor(z_vectors)
    U, S, V = torch.svd(torch.t(z_vectors))
    C = torch.mm(z_vectors, U[:, :2]).tolist()
    C = [x + [labels[i]] for i, x in enumerate(C)]

    df = pd.DataFrame(C, columns=["x", "y", "label"])
    df.head()

    sns.lmplot(x="x", y="y", data=df, fit_reg=False, hue="label")

"""
Save the model incase we need to load it again.
"""
if use_conv:
    torch.save(vae.state_dict(), model_dir + "DCVAE.pt")
else:
    torch.save(vae.state_dict(), model_dir + "VAE.pt")

