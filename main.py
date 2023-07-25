import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train = True
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((512, 512), antialias=True),
    torchvision.transforms.Normalize(0.4456, 0.2618)])

dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")

if torch.backends.mps.is_available():
    dev = torch.device("mps")

if train:
    dataset = torchvision.datasets.Food101(
        root="datasets",
        split="train",
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=16
    )

    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding="same"),
        nn.MaxPool2d(kernel_size=8),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, padding="same"),
        nn.MaxPool2d(kernel_size=8),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=256 * 8 * 8, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=101)
    ).to(dev)

    print(dev)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    epoch_n = 10

    for epoch in range(1, epoch_n + 1):
        for data_in, target in iter(dataloader):
            data_in, target = data_in.to(dev), target.to(dev)
            prediction = model.forward(data_in)

            optimizer.zero_grad()
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            print(float(loss))
