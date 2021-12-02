import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import json
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_tranform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load image
    img_path = "./jly.jpg"
    assert os.path.exists(img_path),"file:{} does not exist.".format(img_path)
    img = Image.open(fp=img_path)

    plt.imshow(img)

    # [N,C,H,W]
    img = data_tranform(img)
    # expand batch dimension
    img = torch.unsqueeze(input=img,dim=0)

    # read class_indict
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(file=json_path,mode="r")
    class_indict = json.load(fp=json_file)

    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(f=weights_path))

    # 测试模式
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        print("output = ",output)
        print("output.shape = ", output.shape)
        predict = torch.softmax(output,dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class:{}  prob:{:.3}".\
        format(class_indict[str(predict_cla)],predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:5}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()























if __name__ == '__main__':
    main()