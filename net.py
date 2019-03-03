import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.full = torch.nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = x.view(-1)
        x = self.full(x)
        return x


EPOCHS = 2

if __name__ == '__main__':
    net = Net()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)

    for epoch in range(EPOCHS):
        for i, data in enumerate(data_loader):
            inputs, labels = data
            optimizer.zero_grad()
            output = net(inputs)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
