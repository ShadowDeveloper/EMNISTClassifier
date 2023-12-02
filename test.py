import time

print("Loading libraries...")
start_time = time.time()

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import random
import datasets
import pickle

print(f"Libraries loaded in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Loading model...")
start_time = time.time()

from model import Model
model = Model().cuda()
model.load_state_dict(torch.load("model.pth"))

print(f"Model loaded in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Loading data...")
start_time = time.time()

X = [(random.randint(-100, 100), random.randint(-100, 100)) for i in range(10000)]
X = np.array(X)

y = [1 if (x > y**2) else 0 for (x, y) in X]

print(X[:10])
print(y[:10])

print(f"Data loaded in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Preparing data...")
start_time = time.time()

X = torch.from_numpy(X).float().cuda()  # Convert to GPU float tensor
y = torch.from_numpy(np.array(y)).float().cuda()  # Convert to GPU float tensor

print(f"Data prepared in {round((time.time() - start_time) * 1000, 3)} ms.")

print("Testing model...")

predictions = []

with torch.no_grad():
    for i in range(len(X)):
        x = X[i]
        y_pred = model(x)
        print(f"Input: {x} | Output: {y_pred} | Expected: {y[i]}")
        predictions.append(y_pred.cpu().numpy()[0])

# graph
plt.scatter([x[0] for x in X.cpu()], [x[1] for x in X.cpu()], c=predictions, cmap='coolwarm_r')
plt.show()

plt.scatter([x[0] for x in X.cpu()], [x[1] for x in X.cpu()], c=y.cpu(), cmap='bwr_r')
plt.show()