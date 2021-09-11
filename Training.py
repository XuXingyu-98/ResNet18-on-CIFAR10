import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
from ResNet import ResNet18

import torchvision.transforms as T

transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

data_dir = './data'

cifar10_train = dset.CIFAR10(data_dir,download=True, transform=transform, train=True)
cifar10_test = dset.CIFAR10(data_dir,download=True, transform=transform, train=False)

loader_train = torch.utils.data.DataLoader(cifar10_train,
                                          batch_size=64,
                                          shuffle=True)

loader_test = torch.utils.data.DataLoader(cifar10_test,
                                          batch_size=64,
                                          shuffle=True)

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print_every = 100


def check_accuracy(loader, model):
    # function for test accuracy on validation and test set

    if False:  # loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return float(acc)


def train_part(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(len(loader_train))
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                # check_accuracy(loader_val, model)
                print()

transform_train  = T.Compose([
        T.RandomCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test  = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

data_dir = './data'

cifar10_train = dset.CIFAR10(data_dir, download=True, transform=transform_train, train=True)
cifar10_test = dset.CIFAR10(data_dir, download=True, transform=transform_test, train=False)
loader_train = torch.utils.data.DataLoader(cifar10_train,
                                          batch_size=64,
                                          shuffle=True)

loader_test = torch.utils.data.DataLoader(cifar10_test,
                                          batch_size=64,
                                          shuffle=True)

model = ResNet18()
optimizer = optim.Adamax(model.parameters(), lr=0.001, weight_decay=1e-7)

train_part(model, optimizer, epochs = 10)


# report test set accuracy

check_accuracy(loader_test, model)


# save the model
torch.save(model.state_dict(), 'model.pt')

import matplotlib.pyplot as plt

plt.tight_layout()

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


vis_labels = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']

for l in vis_labels:
    getattr(model, l).register_forward_hook(get_activation(l))

data, _ = cifar10_test[0]

data = data.unsqueeze_(0).to(device=device, dtype=dtype)

output = model(data)

for idx, l in enumerate(vis_labels):

    act = activation[l].squeeze()

    if idx < 2:
        ncols = 8
    else:
        ncols = 32

    nrows = act.size(0) // ncols

    fig, axarr = plt.subplots(nrows, ncols)
    fig.suptitle(l)

    for i in range(nrows):
        for j in range(ncols):
            axarr[i, j].imshow(act[i * nrows + j].cpu())
            axarr[i, j].axis('off')