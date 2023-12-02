from torch import nn
import torch
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(28*28, 1024)
        self.leakyrelu = nn.LeakyReLU(0.05)
        self.drop = nn.Dropout(0.1)
        self.batchnorm = nn.BatchNorm1d(1024)
        self.lin2 = nn.Linear(1024, 1024)


    def forward(self, x):
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.leakyrelu(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.leakyrelu(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)

        return x

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 512 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            losses.append(loss)

    return losses

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss
