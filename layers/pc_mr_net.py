from time import time
import os
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.optim import Adam

from data.hdf_dataset_loader import HdfDataset
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


def custom_collate_fn(sample):
    ''' This function is only necessary for a batch size greater than 1.
    Since the data samples are of varying sizes and the pytorch data loader class cannot handle this,
    the samples need to be padded to the same size. This enables the use of the inbuilt parallel data loading.
    '''
    shapes = torch.Tensor([s[0].shape for s in sample])
    max_shapes = shapes[:, [2, 3]].max(0)[0]
    input, label = [], []
    for s in sample:
        shape = s[0].shape
        height_padding = int(max_shapes[0] - shape[2])
        width_padding = int(max_shapes[1] - shape[3])
        pad = nn.ZeroPad2d((0, width_padding, 0, height_padding))
        input.append(pad(s[0]))
        label.append(pad(s[1]))
    input = torch.cat(input, 0)
    label = torch.cat(label, 0)
    return input, label


EPOCHS = 20
BATCHSIZE = 10
NUM_WORKERS = 3
MODEL_FILE = "../net_weights/0000.pt"
if __name__ == '__main__':
    t1 = time()
    net = PointCloudMapRegressionNet()
    if os.path.isfile(MODEL_FILE):
        net.load_state_dict(torch.load(MODEL_FILE))
    net = net.cuda()
    optimizer = Adam(net.parameters())
    loss_function = nn.MSELoss()

    train_dataset = HdfDataset("../data/dataset_one_car/train")
    eval_dataset = HdfDataset("../data/dataset_one_car/eval")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True,
                                               collate_fn=custom_collate_fn, num_workers=NUM_WORKERS)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False,
                                              collate_fn=custom_collate_fn, num_workers=NUM_WORKERS)

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
    samples = 0
    for i, (input, labels) in enumerate(eval_loader):
        if i % 10 == 0:
            print(i)
        input = input.cuda()
        labels = labels.cuda()

        output = net(input)
        # mask = labels != 0
        # mask = mask.any(1)
        # mask = mask.unsqueeze(1)
        # mask = torch.cat([mask, mask, mask], 1)
        # average_error_distance=((output[mask] - labels[mask])**2).sum().sqrt()
        predictions = ((output - labels) ** 2).sum(1).sqrt()
        loss += float(predictions.sum())
        # loss += float(loss_function(output, labels))
        samples += len(predictions)
    t3 = time()
    print("Elapsed evalutation time: {}".format(t3 - t2))
    print("Average loss is: ", loss / samples)

    torch.save(net.state_dict(), MODEL_FILE)
