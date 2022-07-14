import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from model import VIN



def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out


def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, float): rep = "%g" % x
    else: rep = str(x)
    return " " * (l - len(rep)) + rep


def get_stats(loss, predictions, labels):
    cp = np.argmax(predictions.cpu().data.numpy(), 1)
    error = np.mean(cp != labels.cpu().data.numpy())
    return loss.item(), error


def print_stats(epoch, avg_loss, avg_error, num_batches, time_duration):
    print(
        fmt_row(10, [
            epoch + 1, avg_loss / num_batches, avg_error / num_batches,
            time_duration
        ]))


def print_header():
    print(fmt_row(10, ["Epoch", "Train Loss", "Train Error", "Epoch Time"]))


class GridworldData(data.Dataset):
    def __init__(self,
                 file,
                 imsize,
                 train=True,
                 transform=None,
                 target_transform=None):
        assert file.endswith('.npz')  # Must be .npz format
        self.file = file
        self.imsize = imsize
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.images, self.S1, self.S2, self.labels =  \
                                self._process(file, self.train)

    def __getitem__(self, index):
        img = self.images[index]
        s1 = self.S1[index]
        s2 = self.S2[index]
        label = self.labels[index]
        # Apply transform if we have one
        if self.transform is not None:
            img = self.transform(img)
        else:  # Internal default transform: Just to Tensor
            img = torch.from_numpy(img)
        # Apply target transform if we have one
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, int(s1), int(s2), int(label)

    def __len__(self):
        return self.images.shape[0]

    def _process(self, file, train):
        """Data format: A list, [train data, test data]
        Each data sample: label, S1, S2, Images, in this order.
        """
        with np.load(file, mmap_mode='r') as f:
            if train:
                images = f['arr_0']
                S1 = f['arr_1']
                S2 = f['arr_2']
                labels = f['arr_3']
            else:
                images = f['arr_4']
                S1 = f['arr_5']
                S2 = f['arr_6']
                labels = f['arr_7']
        # Set proper datatypes
        images = images.astype(np.float32)
        S1 = S1.astype(int)  # (S1, S2) location are integers
        S2 = S2.astype(int)
        labels = labels.astype(int)  # labels are integers
        # Print number of samples
        if train:
            print("Number of Train Samples: {0}".format(images.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(images.shape[0]))
        return images, S1, S2, labels

def train(net, trainloader, config, criterion, optimizer):
    print_header()
    for epoch in range(config.epochs):  # Loop over dataset multiple times
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader):  # Loop over batches of data
            # Get input batch
            X, S1, S2, labels = data
            if X.size()[0] != config.batch_size:
                continue  # Drop those data, if not enough for a batch
            # Automaticlly select device to make the code device agnostic 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            X = X.to(device)
            S1 = S1.to(device)
            S2 = S2.to(device)
            labels = labels.to(device)
            net = net.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs, predictions = net(X, S1, S2, config)
            # Loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update params
            optimizer.step()
            # Calculate Loss and Error
            loss_batch, error_batch = get_stats(loss, predictions, labels)
            avg_loss += loss_batch
            avg_error += error_batch
            num_batches += 1
        time_duration = time.time() - start_time
        # Print epoch logs
        print_stats(epoch, avg_loss, avg_error, num_batches, time_duration)
    print('\nFinished training. \n')

def test(net, testloader, config):
    total, correct = 0.0, 0.0
    for i, data in enumerate(testloader):
        # Get inputs
        X, S1, S2, labels = data
        if X.size()[0] != config.batch_size:
            continue  # Drop those data, if not enough for a batch
        # automaticlly select device, device agnostic 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = X.to(device)
        S1 = S1.to(device)
        S2 = S2.to(device)
        labels = labels.to(device)
        net = net.to(device)
        # Forward pass
        outputs, predictions = net(X, S1, S2, config)
        # Select actions with max scores(logits)
        _, predicted = torch.max(outputs, dim=1, keepdim=True)
        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data
        # Compute test accuracy
        correct += (torch.eq(torch.squeeze(predicted), labels)).sum()
        total += labels.size()[0]
    print('Test Accuracy: {:.2f}%'.format(100 * (correct / total)))

if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datafile',
        type=str,
        default='dataset/gridworld_8x8.npz',
        help='Path to data file')
    parser.add_argument('--imsize', type=int, default=8, help='Size of image')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.005,
        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument(
        '--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument(
        '--k', type=int, default=10, help='Number of Value Iterations')
    parser.add_argument(
        '--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument(
        '--l_h',
        type=int,
        default=150,
        help='Number of channels in first hidden layer')
    parser.add_argument(
        '--l_q',
        type=int,
        default=10,
        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    config = parser.parse_args()
    # Get path to save trained model
    save_path = "trained/vin_{0}x{0}.pth".format(config.imsize)
    # Instantiate a VIN model
    net = VIN(config)
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=config.lr, eps=1e-6)
    # Dataset transformer: torchvision.transforms
    transform = None
    # Define Dataset
    trainset = GridworldData(
        config.datafile, imsize=config.imsize, train=True, transform=transform)
    testset = GridworldData(
        config.datafile,
        imsize=config.imsize,
        train=False,
        transform=transform)
    # Create Dataloader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    # Train the model
    train(net, trainloader, config, criterion, optimizer)
    # Test accuracy
    test(net, testloader, config)
    # Save the trained model parameters
    torch.save(net.state_dict(), save_path)
