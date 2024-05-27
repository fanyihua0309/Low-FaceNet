import os
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm


def synthesize(path):
    # 读取正常光照图像
    normal_image = cv2.imread(path)

    # 定义伽马值（小于1表示降低亮度，大于1表示增加亮度）
    gamma = random.uniform(2.15, 2.25)  # 调整伽马值以模拟低光照

    # 应用伽马校正
    low_light_image = np.power(normal_image / 255.0, gamma)
    low_light_image = (low_light_image * 255).astype(np.uint8)

    # 添加噪声
    mean = 0  # 噪声均值
    stddev = random.uniform(0.26, 0.30)  # 噪声标准差
    noise = np.random.normal(mean, stddev, low_light_image.shape).astype('uint8')
    low_light_image = cv2.add(low_light_image, noise)

    return low_light_image


def pipeline():
    root = r'C:\Users\fyh\Desktop\My-Face-Dataset\test'
    dir_path = os.path.join(root, 'CelebA-Test', 'normal')
    save_dir_path = os.path.join(root, 'CelebA-Test-2', 'underexposed')
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    with tqdm(total=len(os.listdir(dir_path))) as pbar:
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            output = synthesize(img_path)
            save_path = os.path.join(save_dir_path, img_name)
            cv2.imwrite(save_path, output)
            pbar.update(1)


if __name__ == '__main__':
    pipeline()
