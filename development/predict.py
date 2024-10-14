import time

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import one_hot
import torch
import development.gen_ImageCaptcha as gen_ImageCaptcha
from torchvision import transforms
from model import mydatasets, MyModel


def test_pred():
    m = MyModel()
    m.load_state_dict(torch.load("../deploy/model/22_0.0007106820558649762.pth"))
    m.eval()
    test_data = mydatasets("../dataset/test")

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_length = test_data.__len__()

    correct = 0
    spendtime = []
    for i, (imgs, lables) in enumerate(test_dataloader):
        imgs = imgs
        lables = lables
        lables = lables.view(-1, gen_ImageCaptcha.captcha_array.__len__())
        lables_text = one_hot.vectotext(lables)
        start_time = time.time()
        predict_outputs = m(imgs)
        predict_outputs = predict_outputs.view(-1, gen_ImageCaptcha.captcha_array.__len__())
        predict_labels = one_hot.vectotext(predict_outputs)
        spendtime.append(time.time() - start_time)
        print(predict_labels, lables_text)
        if predict_labels == lables_text:
            correct += 1
        else:
            pass

    correct_ = correct / test_length * 100
    data = np.array(spendtime)
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(f"verify spend time, average spend: {np.mean(spendtime)} and Success rate: {correct_}")
    plt.ylabel("time")
    # 显示图像
    plt.show()


if __name__ == '__main__':
    test_pred()
