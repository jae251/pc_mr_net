from time import time

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.optim import Adam

from data.hdf_dataset_loader import HdfLoader
from layers.convolution_module import FirstLayer, ConvolutionModule


class PointCloudMapRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FirstLayer()
        self.conv2 = ConvolutionModule(40, 20)
        self.conv2 = ConvolutionModule(40, 20)
        self.conv3 = ConvolutionModule(40, 20)
        self.conv4 = nn.Conv2d(40, 3, 1)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = self.conv4(x)
        return x


EPOCHS = 20
if __name__ == '__main__':
    t1 = time()
    net = PointCloudMapRegressionNet()
    net = net.cuda()
    optimizer = Adam(net.parameters())
    loss_function = nn.MSELoss()

    train_loader = HdfLoader("../data/dataset_one_car/train", shuffle=True)
    eval_loader = HdfLoader("../data/dataset_one_car/eval", shuffle=False)

    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(eval_loader)))

    # training loop
    for epoch in range(EPOCHS):
        print("EPOCH ", epoch)
        for i, (input, labels) in enumerate(train_loader):
            if i % 10 == 0:
                print(i)
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
    loss = 0
    for i, (input, labels) in enumerate(eval_loader):
        if i % 10 == 0:
            print(i)
        input = input.cuda()
        labels = labels.cuda()

        output = net(input)
        # mask = labels != 0
        # mask = mask.any(1)
        loss += float(loss_function(output, labels))
    t3 = time()
    print("Elapsed evalutation time: {}".format(t3 - t2))
    print("Average loss is: ", loss / i)

# torch.save(model.state_dict(), model.name())
