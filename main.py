import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import learning_time_est as lte

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((128, 128), antialias=True),
    torchvision.transforms.Normalize(0.4456, 0.2618)])

dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")

if torch.backends.mps.is_available():
    dev = torch.device("mps")

train = True
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
        batch_size=64
    )

    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=101, kernel_size=5, padding="same"),
        nn.MaxPool2d(kernel_size=8),
        nn.ReLU(),
        nn.Conv2d(in_channels=101, out_channels=101, kernel_size=5, padding="same"),
        nn.MaxPool2d(kernel_size=4),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=101 * 4 * 4, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=101)
    ).to(dev)

    print(dev)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    epoch_n = 2

    t0 = time.time()
    for epoch in range(1, epoch_n + 1):
        loss_sum = 0
        for data_in, target in iter(dataloader):
            data_in, target = data_in.to(dev), target.to(dev)
            prediction = model.forward(data_in)

            optimizer.zero_grad()
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

        loss_sum += float(loss) / len(dataloader)
        print(f"mean loss: {loss_sum}")
        lte.show_time(t0, epoch / epoch_n)

    torch.save(obj=model, f="model.pth")
    print("Model saved.")

else:
    dataset = torchvision.datasets.Food101(
        root="datasets",
        split="test",
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=16
    )

    model = torch.load(f="model.pth").to(dev)
    print(dev)
    print(model)

    good = 0
    all = 0
    for data_in, target in iter(dataloader):
        data_in, target = data_in.to(dev), target.to(dev)
        prediction = model.forward(data_in)

        for i in range(len(data_in)):
            if torch.argmax(prediction) == target:
                good += 1

            all += 1
    
    print(f"Accuracy: {good / all}")
