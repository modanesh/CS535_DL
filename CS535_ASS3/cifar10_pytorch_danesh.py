from __future__ import print_function
from __future__ import division

import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        # TODO: Task 4 - tune the network - adding more layers
        # self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        # self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(128 * 4 * 4, 512)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)

        # TODO: Task 1 - add batchnorm layer and check for improvements
        # self.batch = nn.BatchNorm1d(512)

        # TODO: Task 2 - add a new fc layer
        # self.fc1_5 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # TODO: Task 4 - tune the network - adding more layers
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = self.pool(x)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))

        # TODO: Task 1 - add batchnorm layer and check for improvements
        # x = self.batch(x)

        # TODO: Task 4 - tune the network - adding dropout regularization
        # x = F.dropout(x, 0.5)

        # TODO: Task 2 - add a new fc layer
        # x = F.relu(self.fc1_5(x))

        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # correct += (predicted == labels.data).sum()
        for i in range(len(predicted)):
            if predicted[i] == labels.data[i]:
                correct += 1

        loss = criterion(outputs, labels)
        total_loss += loss.data
    net.train() # Why would I do this?
    return total_loss / total, correct / total


def test_model(model_path):
    loaded_model = torch.load(model_path)
    print(loaded_model.keys())

    return loaded_model


def plot_diagrams(train_results, test_results, num_epochs, phase, task):
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_results, '-b', label="%s %s" % ('Train', phase))
    plt.plot(range(1, num_epochs + 1), test_results, '-g', label="%s %s" % ('Test', phase))
    plt.xlim(1, num_epochs)
    # plt.ylim(0, 1)
    max_ylim = max(test_results)
    plt.ylim(0, math.ceil(max_ylim))
    plt.xlabel("Epoch")
    plt.ylabel(phase)
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.title("%s of train and test" % phase)
    plt.savefig("%s_settings_%d.png" % (phase, task))


if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 10 #maximum epoch to train
    model_path = 'mytraining.pth'
    train_loss_set, train_acc_set, test_loss_set, test_acc_set = [], [], [], []

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model')
    net = Net().cuda()
    net.train() # Why would I do this?

    # TODO: Task 2 - load partial model
    # pre_trained_dict = torch.load(model_path)
    # current_trained_dict = net.state_dict()
    # for key in current_trained_dict:
    #     if key in pre_trained_dict and key != 'fc2.weight' and key != 'fc2.bias':
    #         current_trained_dict[key] = pre_trained_dict[key]
    # net.load_state_dict(current_trained_dict)




    criterion = nn.CrossEntropyLoss()
    # TODO: Task 3 - check different optimization algorithms
    # TODO: RMSprop and Adam loss are weird. Need to double check.
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99)
    # optimizer = optim.Adagrad(net.parameters(), lr=0.01)
    # optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))


    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        train_loss_set.append(train_loss)
        train_acc_set.append(train_acc)
        test_loss_set.append(test_loss)
        test_acc_set.append(test_acc)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), model_path)

    # plotting the diagrams for each task
    plot_diagrams(train_loss_set, test_loss_set, MAX_EPOCH, 'Loss', 1111)
    plot_diagrams(train_acc_set, test_acc_set, MAX_EPOCH, 'Accuracy', 1111)