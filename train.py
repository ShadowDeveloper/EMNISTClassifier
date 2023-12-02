import time

print("Loading libraries...")
start_time = time.time()

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import *
import numpy as np
import matplotlib.pyplot as plt

print(f"Libraries loaded in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Loading data...")

training_data = datasets.EMNIST(root="data", train=True, download=True, transform=ToTensor(), split="bymerge")

test_data = datasets.EMNIST(root="data", train=False, download=True, transform=ToTensor(), split="bymerge")

print(f"Data loaded in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Setting configuration...")
start_time = time.time()

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"Configuration set in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Loading model...")
start_time = time.time()


model = Model().cuda()

print(f"Model loaded in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Preparing data...")
start_time = time.time()

# Create data loaders.
batch_size = 128

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

'''for i in range(len(training_data)):
    x = training_data[i]
    y = test_data[i]
    print(f"Input: {x} | Output: {y}")'''

print(f"Data prepared in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Training model...")
start_time = time.time()

train_losses = []
test_losses = []
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 16

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    test_loss = test(test_dataloader, model, loss_fn, device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)


print(f"Model trained in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Saving model...")
start_time = time.time()

torch.save(model.state_dict(), "model.pth")

print(f"Model saved in {round((time.time() - start_time) * 1000, 3)} ms.")

'''print("Visualizing loss...")

plt.plot(train_losses, label="Training loss", c="#EE6660")
plt.plot(test_losses, label="Test loss", c="#90DDF0")
plt.legend()
plt.show()

print(f"Loss visualized.")'''