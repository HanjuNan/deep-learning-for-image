import torch
import torch.nn as nn
import torch.nn.functional as F

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.fc1 = nn.Linear(in_features=32*5*5,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=10)

    def forward(self,x):
        x = F.relu(self.conv1(x))   # input(3,32,32) output(16,28,28)
        x = self.pool1(x)  # output(16,14,14)
        x = F.relu(self.conv2(x))  # output(32,10,10)
        x = self.pool2(x)  # output(32,5,5)
        x = x.view(-1,32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)
        # 注意这里最后没有使用softmax激活
        return x

if __name__ == '__main__':
    input1 = torch.rand([32,3,32,32])
    model = Lenet()
    output1 = model(input1)
    print("output1.shape = ",output1.shape)
