import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self,num_classes=1000,init_weight=False):
        super(AlexNet, self).__init__()

        # 特征提取网络
        self.features = nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2), # input[3,224,224] output[48,55,55]
            # 不要忘记卷积完要池化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),  # output[48,27,27]
            nn.Conv2d(48,128,kernel_size=5,padding=2),  # output[128,27,27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),  # output[128,13,13]
            nn.Conv2d(128,192,kernel_size=3,padding=1),  # output[192,13,13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192,13,13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3,padding=1),  # output[128,13,13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),  # output[128,6,6]
        )

        # 分类网络
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,num_classes)
        )

        # 初始化权重
        if init_weight:
            self._initialize_weights()
    def forward(self,x):

        x = self.features(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.01)
                nn.init.constant_(m.bias,0)


if __name__ == '__main__':
    alexnet = AlexNet(num_classes=5)
    inputs = torch.rand(size=(10,3,224,224))
    outputs = alexnet(inputs)
    print("outputs.shape = ",outputs.shape)

