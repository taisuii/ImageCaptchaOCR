import torch
from matplotlib import pyplot as plt
from torch import nn
import development.gen_ImageCaptcha as gen_ImageCaptcha
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import one_hot
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class mydatasets(Dataset):
    def __init__(self, root_dir):
        super(mydatasets, self).__init__()
        self.list_image_path = [os.path.join(root_dir, image_name) for image_name in os.listdir(root_dir)]
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image_path = self.list_image_path[index]
        img_ = Image.open(image_path)
        image_name = image_path.split("\\")[-1]
        img_tesor = self.transforms(img_)
        img_lable = image_name.split(".")[0]
        img_lable = one_hot.text2vec(img_lable)
        img_lable = img_lable.view(1, -1)[0]
        return img_tesor, img_lable

    def __len__(self):
        return self.list_image_path.__len__()


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 添加Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 使用全局平均池化代替Flatten
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 4096),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(4096, gen_ImageCaptcha.captcha_size * len(gen_ImageCaptcha.captcha_array))
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)  # 全局平均池化
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


def train(epoch):
    train_datas = mydatasets("../dataset/train")
    train_dataloader = DataLoader(train_datas, batch_size=64, shuffle=True)
    m = MyModel().cuda()

    loss_fn = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    epoch_losses = []
    for i in range(epoch):
        losses = []
        # 迭代器进度条
        data_loader_tqdm = tqdm(train_dataloader)

        epoch_loss = 0
        for inputs, labels in data_loader_tqdm:
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            outputs = m(inputs)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            epoch_loss = np.mean(losses)
            data_loader_tqdm.set_description(
                f"This epoch is {str(i + 1)} and it's loss is {loss.item()}, average loss {epoch_loss}"
            )

            loss.backward()
            optimizer.step()
        epoch_losses.append(epoch_loss)
        # 每过一个batch就保存一次模型
        torch.save(m.state_dict(), f'../deploy/model/{str(i + 1)}_{epoch_loss}.pth')

    # loss 变化绘制代码
    data = np.array(epoch_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(f"{epoch} epoch loss change")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    # 显示图像
    plt.show()
    print(f"completed. Model saved.")

if __name__ == '__main__':
    train(24)
