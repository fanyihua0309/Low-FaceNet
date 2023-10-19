import os
import shutil
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import torchvision
from model import EnhanceNet
from utils.general import get_mode


def enhance_images(underexposed_path, enhanced_path, enhance_checkpoint):
    if os.path.exists(enhanced_path):
        shutil.rmtree(enhanced_path)
    if not os.path.exists(enhanced_path):
        os.makedirs(enhanced_path)
    print(f'Load enhance model from {enhance_checkpoint}...')
    model = EnhanceNet(phase='test').cuda()
    model.load_state_dict(torch.load(enhance_checkpoint))

    def enhance_single_image(model, dark_img, result_path):
        dark_img = (np.asarray(dark_img) / 255.0)
        dark_img = torch.from_numpy(dark_img).float()
        dark_img = dark_img.permute(2, 0, 1)
        dark_img = dark_img.cuda().unsqueeze(0)
        enhanced_image = model(dark_img)
        torchvision.utils.save_image(enhanced_image, result_path, normalizer=True)

    mode = get_mode(underexposed_path)
    model.eval()
    with tqdm(total=len(os.listdir(underexposed_path)), desc=f'Enhance', postfix=dict, mininterval=0.3) as pbar:
        if mode == 'isfile':
            for index, img_name in enumerate(os.listdir(underexposed_path)):
                img_path = os.path.join(underexposed_path, img_name)
                img = Image.open(img_path)
                result_path = os.path.join(enhanced_path, img_name)
                enhance_single_image(model, img, result_path)
                pbar.update(1)
        elif mode == 'isdir':
            for index, identity in enumerate(os.listdir(underexposed_path)):
                dir_path = os.path.join(underexposed_path, identity)
                save_dir_path = os.path.join(enhanced_path, identity)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                for i, img_name in enumerate(os.listdir(dir_path)):
                    img_path = os.path.join(dir_path, img_name)
                    img = Image.open(img_path)
                    result_path = os.path.join(save_dir_path, img_name)
                    enhance_single_image(model, img, result_path)
                pbar.update(1)