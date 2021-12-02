import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

from torch.utils.data import DataLoader
from torchvision import datasets
import json
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from model import AlexNet
from tqdm import tqdm

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print("using {} device. ".format(device))

    data_transform = {
        "train":transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ]),
        "val":transforms.Compose([
            transforms.Resize((224,224)), # cannot 224, must (224, 224)
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    }


    data_root = os.path.abspath(os.path.join(os.getcwd(),"../../")) # get data root path
    # print("os.getcwd() = ",os.getcwd())
    # print("os.join = ",os.path.join(os.getcwd(),"../../"))
    # print("data_root = ",data_root)
    # data_root = C:\Users\Administrator\Desktop\deep-learning-for-image
    image_path = os.path.join(data_root,"data_set","starts_data") # flower data set path
    assert os.path.exists(image_path),"{} path does not exist.".format(image_path)

    train_path = os.path.join(image_path,"train")
    # print("image_path = ",image_path)
    # print("train_path = ",train_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,"train"),transform=data_transform["train"])
    train_num = len(train_dataset)
    # print("train_num = ",train_num)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    # print("flower_list = ",flower_list)
    cla_dict = dict((val,key) for key,val in flower_list.items())


    # cla_dict = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    # print("cla_dict = ",cla_dict)

    # for data in flower_list.items():
    #     print("data = ",data)
    #     data = ('daisy', 0)
    #     data = ('dandelion', 1)
    #     data = ('roses', 2)
    #     data = ('sunflowers', 3)
    #     data = ('tulips', 4)


    # write dict into json file
    json_str = json.dumps(obj=cla_dict,indent=4)
    with open(file="class_indices.json",mode="w") as json_file:
        json_file.write(json_str)

    batch_size = 32

    # nw = min([os.cpu_count(),batch_size if batch_size > 1 else 0,8]) # number of workers
    nw = 0
    # print("nw = ",nw)
    # print("os.cpu_count() = ",os.cpu_count())
    # print("[os.cpu_count(),batch_size if batch_size > 1 else 0,8] = ",[os.cpu_count(),batch_size if batch_size > 1 else 0,8])
    # print("Using {} dataloader workers every process".format(nw))

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=nw)

    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path,"val"),transform=data_transform["val"])
    # val_num = len(validate_dataset)
    # # print("val_num = ",val_num)
    # validate_loader = DataLoader(dataset=validate_dataset,batch_size=4,shuffle=False,num_workers=nw,drop_last=True)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    test_data_iter = iter(validate_loader)
    test_image, test_label = test_data_iter.next()

    # def imshow(img):
    #     img = img / 2 + 0.5  #unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(a=npimg,axes=(1,2,0)))
    #     plt.show()
    # print(" ".join("%5s " % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(torchvision.utils.make_grid(tensor=test_image))

    # 定义网络
    net = AlexNet(num_classes=2,init_weight=True)
    net.to(device)
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(),lr=0.0002)

    epochs = 10
    save_path = "./AlexNet.pth"
    best_acc = 0.0
    train_steps = len(train_loader)
    # 循环迭代训练
    for epoch in range(epochs):
        # train模式
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step,data in enumerate(train_bar):
            images,labels = data
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = net(images.to(device))
            # 计算损失
            loss = loss_function(outputs,labels.to(device))
            # 误差返现传播
            loss.backward()
            # 优化器更新参数
            optimizer.step()

            # 打印统计信息
            # 一次epoch的损失
            running_loss += loss.item()

            train_bar.desc = "training epoch[{}/{}] loss:{:.3f}".format(epoch+1,step+1,loss)


        # validate模式
        # 每次训练一个epoch就验证一下模型
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        # 验证阶段不需要计算梯度
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_image,val_labels = val_data
                # 模型推理
                outputs = net(val_image.to(device))
                # 预测下标
                predict_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(predict_y,val_labels.to(device)).sum().item()

            val_accurate = acc / val_num
            print("[epoch %d] train_loss: %.3f val_accuracy: %.3f"
                  % (epoch+1,running_loss / train_steps,val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(obj=net.state_dict(),f=save_path)

    print("Finished Training")


if __name__ == '__main__':
    main()

