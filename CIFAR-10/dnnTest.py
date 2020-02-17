import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # functions to show an image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 4 * 4, 150)
        self.fc2 = nn.Linear(150, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    use_cuda = True
    batch_size = 64
    num_workers = 2
    train_again = True
    LR = 0.001
    Momentum = 0.9
    epoch_num = 30
    showImage = False
    # 归一化
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 导入测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if train_again:
        # 导入训练集
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        if use_cuda:
            net = Net().cuda()
        else:
            net = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

        start_time = time.time()
        for epoch in range(epoch_num):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, (x, y) in enumerate(trainloader, 0):

                if use_cuda:
                    inputs = x.cuda()
                    labels = y.cuda()
                else:
                    inputs = x
                    labels = y

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss/batch_size*100))
                    running_loss = 0.0
        end_time = time.time()
        print('Finished Training')
        print('Training Time:', end_time-start_time)

        torch.save(net, 'model.pkl')
    else:
        net = torch.load('model.pkl')

    net.cpu()
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    if showImage:
        # print images
        imshow(torchvision.utils.make_grid(images.cpu()))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():  # 数据不计算梯度，不需要反向传播
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
