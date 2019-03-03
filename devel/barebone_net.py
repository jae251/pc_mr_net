from time import time

import torch
import torch.nn as nn
from torch.nn.functional import relu, log_softmax
from torch.optim import Adam

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MapRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.conv2 = nn.Conv2d(20, 20, 3)
        self.conv3 = nn.Conv2d(20, 2, 1)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        return x


class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(28 * 28, 200)
        self.lin2 = nn.Linear(200, 200)
        self.lin3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = relu(self.lin1(x))
        x = relu(self.lin2(x))
        x = log_softmax(self.lin3(x))
        return x


EPOCHS = 10
BATCHSIZE = 100
if __name__ == '__main__':
    t1 = time()
    net = LinearNet()
    net = net.cuda()
    optimizer = Adam(net.parameters())
    loss_function = nn.CrossEntropyLoss()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_data = MNIST("./data", train=True, transform=tf, download=True)
    eval_data = MNIST("./data", train=False, transform=tf, download=True)

    train_loader = DataLoader(train_data, BATCHSIZE, shuffle=True)
    eval_loader = DataLoader(eval_data, BATCHSIZE, shuffle=False)

    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(eval_loader)))

    # training loop
    for epoch in range(EPOCHS):
        for i, (input, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            input = input.cuda()
            labels = labels.cuda()

            output = net(input)

            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
    t2 = time()
    print("Elapsed training time: {}".format(t2 - t1))

    # evaluation loop
    total_inputs = 0
    correct_labels = 0
    for i, (input, labels) in enumerate(eval_loader):
        input = input.cuda()
        labels = labels.cuda()

        output = net(input)

        _, predictions = torch.max(output.data, 1)
        total_inputs += input.data.size()[0]
        correct_labels += (predictions == labels.data).sum()
    t3 = time()
    print("Elapsed evalutaion time: {}".format(t3 - t2))
    print("Accuracy is {:.2f} %".format(correct_labels * 100.0 / total_inputs))

# torch.save(model.state_dict(), model.name())
