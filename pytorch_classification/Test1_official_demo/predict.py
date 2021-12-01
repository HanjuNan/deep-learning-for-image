import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Lenet

def main():
    transform = transforms.Compose([
        # 缩放图片
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ])
    # 类别元组
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 创建网络模型
    net = Lenet()
    net.load_state_dict(torch.load(f="./Lenet.pth"))

    # 打开图片
    im = Image.open("1.jpg")
    # 对图片进行转换
    im = transform(im)
    # 因为pytorch需要的Tensor类型是BCHW的
    # 对数据进行扩展一个维度
    im = torch.unsqueeze(im,dim=0)

    # 不需要计算梯度
    with torch.no_grad():
        outputs = net(im)
        # 预测类别的索引
        predict = torch.max(outputs,dim=1)[1].data.numpy()
    # 预测的类别是
    # 图片最终被预测为: truck
    print("图片最终被预测为: ",classes[int(predict)])
    


if __name__ == '__main__':
    main()