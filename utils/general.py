import logging
import os
import random
import sys
import torch
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms
from seg_module import deeplabv3plus_mobilenet


def log_config(file_path):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')
    fh = logging.FileHandler(file_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return logging


def get_mode(path):
    test_path = os.path.join(path, os.listdir(path)[0])
    if os.path.isfile(test_path):
        mode = 'isfile'
    elif os.path.isdir(test_path):
        mode = 'isdir'
    return mode


# 计算图片的平均亮度值
def get_average_illumination(img):
    i = 0.299 * torch.mean(img[:, 0, :, :]) + 0.587 * torch.mean(img[:, 1, :, :]) + 0.114 * torch.mean(img[:, 2, :, :])
    return i.item()


# 初始化网络参数
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 加载对比学习样本
def load_samples(path, num, crop_size):
    samples = random.sample(os.listdir(path), num)
    trans_tensor = transforms.ToTensor()
    list = []
    for sample in samples:
        img = Image.open(os.path.join(path, sample))
        img = img.resize((crop_size, crop_size), Image.ANTIALIAS)
        img = trans_tensor(img)
        list.append(img)
    res = torch.stack(list, dim=0).cuda()
    return res


# 对语义分割网络的输入做变换
def transform_seg_input(input):
    trans_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    input1, input2 = input.split(1, dim=0)
    input1 = input1.squeeze()
    input2 = input2.squeeze()
    input1 = trans_norm(input1)
    input2 = trans_norm(input2)
    trans_input = torch.stack([input1, input2], dim=0)
    return trans_input


# 加载语义分割模型
def load_seg(seg_checkpoint_path):
    # 语义分割类别数
    seg_model = deeplabv3plus_mobilenet(num_classes=11, output_stride=16)
    checkpoint = torch.load(seg_checkpoint_path, map_location=torch.device('cpu'))
    seg_model.load_state_dict(checkpoint['model_state'])
    seg_model = torch.nn.DataParallel(seg_model)
    seg_model = seg_model.cuda()
    return seg_model


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = momentum


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
