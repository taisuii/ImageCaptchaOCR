@[TOC](pytorch_CNN英数验证码识别模型训练)

### 安卓逆向，JS逆向，图像识别，在线接单，全套源码+部署+算法联系QQ: 27788854，wechat: taisuivip，[telegram: rtais00](https://t.me/rtais00)
### 微信公众号：R逆向
# 验证码样式
#### 问题案例：
&nbsp;&nbsp;&nbsp;&nbsp;项目github：[https://github.com/taisuii/ImageCaptchaOCR](https://github.com/taisuii/ImageCaptchaOCR)
&nbsp;&nbsp;&nbsp;&nbsp;验证码样式通常为通常为英文数字
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/93c57be2e23949988de79c8452341434.png#pic_center)
#### 解决方案或思路：
&nbsp;&nbsp;&nbsp;&nbsp;卷积神经网络，搭建模型，使其输出为5*(26+26+10)=310的张量，解码为英文和数字
# 数据集准备
#### 整理数据集：
&nbsp;&nbsp;&nbsp;&nbsp;先关注公众号：R逆向，回复：验证码数据集
&nbsp;&nbsp;&nbsp;&nbsp;用两个文件夹，train存放数据集，test存放测试数据集，内容如下，标签就是图片名字
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/347c6358daf2494d95ddb9cbcf90838a.png#pic_center)
#### 自定义数据集：
```python
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
```
# 模型训练
#### 模型代码如下：
&nbsp;&nbsp;&nbsp;&nbsp;随便写几个层，留意输入和输出，gen_ImageCaptcha.captcha_size * len(gen_ImageCaptcha.captcha_array)就是5*(26+26+10)
```python
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
```
#### 模型训练代码
&nbsp;&nbsp;&nbsp;&nbsp;这里的代码很常规，分类模型的训练，这里训练24个epoch
```python
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
        torch.save(m.state_dict(), f'../deplo/model/{str(i + 1)}_{epoch_loss}.pth')


if __name__ == '__main__':
    train(24)

```
loss值变化如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e155e98f50d24c5a9b244c92c9e98956.png#pic_center)

# 调用
#### 模型预测：
&nbsp;&nbsp;&nbsp;&nbsp;把模型的输出转换成字符，也就是每隔62就解码一个字符
```python
def test_pred():
    m = MyModel()
    m.load_state_dict(torch.load("../deploy/model/22_0.0007106820558649762.pth"))
    m.eval()
    test_data = mydatasets("../dataset/test")

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_length = test_data.__len__()

    correct = 0

    for i, (imgs, lables) in enumerate(test_dataloader):
        imgs = imgs
        lables = lables
        lables = lables.view(-1, gen_ImageCaptcha.captcha_array.__len__())
        lables_text = one_hot.vectotext(lables)
        start_time = time.time()
        predict_outputs = m(imgs)
        predict_outputs = predict_outputs.view(-1, gen_ImageCaptcha.captcha_array.__len__())
        predict_labels = one_hot.vectotext(predict_outputs)
        print(time.time() - start_time)
        if predict_labels == lables_text:
            correct += 1
        else:
            pass
    print("正确率{}".format(correct / test_length * 100))


if __name__ == '__main__':
    test_pred()
```
#### 识别速度和平均成功率如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6c81227c1ecb4f9e81f4eb564d5b3b85.png#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fcff4e72746c43148ffb16a3b1729f74.png#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6f4eeb23947542f38c7077563ba9ffe3.jpeg#pic_center)
