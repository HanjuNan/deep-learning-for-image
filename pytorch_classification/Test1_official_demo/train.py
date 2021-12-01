import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import Lenet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():

    # 对图像做变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ])
    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root="./data",train=True,transform=transform,download=True)
    # 定义训练集的dataloader
    train_loader = DataLoader(dataset=train_set,batch_size=36,shuffle=True,num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root="./data",train=False,transform=transform,download=True)
    val_loader = DataLoader(dataset=val_set,batch_size=5000,shuffle=True,num_workers=0)

    # 查看验证集张啥样子
    val_data_iter = iter(val_loader)
    val_image,val_label = val_data_iter.next()

    img0,lb0 = val_image[0],val_label[0]
    print("img0.shape = ",img0.shape)
    print("lb0 = ",lb0)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # # show images
    # imshow(torchvision.utils.make_grid(val_image))
    # # print labels
    # print(' '.join('%5s' % classes[val_label[j]] for j in range(4)))

    # 定义网络
    net = Lenet()

    # 定义多分类的交叉熵损失函数
    # Note that this case is equivalent to the combination of:
    # class: `~torch.nn.LogSoftmax` and: class: `~torch.nn.NLLLoss`.
    # 意思是交叉熵损失函数包括了LogSoftmax，所以不需要在网络最后一层再Softmax了
    loss_function = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = torch.optim.Adam(params=net.parameters(),lr=0.001)

    # 循环训练5轮
    for epoch in range(5):
        # 一个批次的损失
        running_loss = 0.0
        for step,data in enumerate(train_loader,start=0):
            inputs,labels = data

            # 梯度清零
            optimizer.zero_grad()
            # 前向传播,得到预测结果
            outputs = net(inputs)
            # 使用交叉熵函数计算损失，该损失为一个batch的损失
            loss = loss_function(outputs,labels)
            # 误差反向传播
            loss.backward()
            # 优化器更新参数
            optimizer.step()

            # 打印信息
            running_loss += loss.item()
            # 每500个batch打印一次
            if step % 500 == 499:
                # 测试时不计算梯度，减少资源消耗
                with torch.no_grad():
                    outputs = net(val_image) # [batch, 10]
                    predict_y = torch.max(outputs,dim=1)[1]
                    accuracy = torch.eq(predict_y,val_label).sum().item() / val_label.size(0)

                    print("[%d,%5d] train_loss:%.3f test_accuracy:%.3f" %
                          (epoch+1,step+1,running_loss/500,accuracy))


    print("Finished Training")

    # 保存模型
    save_path = "./Lenet.pth"
    # obj:要保存的对象，f:要保存的路径
    torch.save(obj=net.state_dict(),f=save_path)

if __name__ == '__main__':
    main()