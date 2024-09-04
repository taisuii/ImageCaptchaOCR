import development.gen_ImageCaptcha as gen_ImageCaptcha
import torch
import torch.nn.functional as F
def text2vec(text):
    vectors=torch.zeros((gen_ImageCaptcha.captcha_size,gen_ImageCaptcha.captcha_array.__len__()))
    for i in range(len(text)):
        vectors[i,gen_ImageCaptcha.captcha_array.index(text[i])]=1
    return vectors
def vectotext(vec):

    vec=torch.argmax(vec,dim=1)

    text_label=""
    for v in vec:
        text_label+=gen_ImageCaptcha.captcha_array[v]
    return  text_label

if __name__ == '__main__':
    vec=text2vec("aaab")
    print(vec.shape)


    print(vectotext(vec))