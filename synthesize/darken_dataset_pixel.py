import random
import os
import time
from PIL import Image
import torch
import torchvision


# 计算图片的平均亮度值
def get_average_illumination(img):
    trans = torchvision.transforms.ToTensor()
    img = trans(img)
    i = 0.299 * torch.mean(img[0, :, :]) + 0.587 * torch.mean(img[1, :, :]) + 0.114 * torch.mean(img[2, :, :])
    return i.item()


# 暗化处理训练数据集
def darken_process(from_path, to_path, darken_ratio):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] Darken face dataset start.')
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    cnt = 0
    sum_ill = 0
    for img_name in os.listdir(from_path):
        img_path = os.path.join(from_path, img_name)
        img = Image.open(img_path)
        # 图片有尺寸限制
        # if img.size < (384, 384):
        #     continue
        # 对图片进行变暗处理
        param = random.uniform(darken_ratio[0], darken_ratio[1])
        dark_img = img.point(lambda p: p * param)
        dark_img_path = os.path.join(to_path, img_name)
        dark_img.save(dark_img_path)
        cnt += 1
        sum_ill += get_average_illumination(dark_img)
        ill = sum_ill / cnt
        print('\r' '%.2f%%   index=%d/%d   param=%f   aver_illumination=%f' % (
            cnt / len(os.listdir(from_path)) * 100, cnt, len(os.listdir(from_path)), param, ill), end='')
    print(f'\n[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] Darken face dataset end.')


if __name__ == '__main__':
    from_path = r'C:\Users\fyh\Desktop\experiment\input'
    to_path = r'C:\Users\fyh\Desktop\experiment\output'
    darken_ratio = [0.35, 0.45]
    darken_process(from_path, to_path, darken_ratio)
