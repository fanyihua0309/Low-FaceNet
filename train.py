import torch
from torch.utils import data
import argparse
import numpy as np
import random
import os

from model import EnhanceNet
from face_module.retinaface import Retinaface
from dataset import Face_dataset

from loss.smooth_loss import SmoothLoss
from loss.contrast_loss import ContrastLoss, ContrastBrightnessLoss
from loss.color_loss import ColorLoss
from loss.losses import PerceptualLoss, SegBrightnessLoss, ColorConsistencyLoss

from utils.face import load_face_database, face_encoding
from utils.general import log_config, get_average_illumination, load_samples, init_weights


def get_argparser():
    start_epoch = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--start_epoch', default=start_epoch, type=int)
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--crop_size', default=384, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--dataset_name', default='LaPa-Face', type=str)
    parser.add_argument('--contrast_positive_path', default='dataset/Contrast/high', type=str)
    parser.add_argument('--contrast_negative_path', default='dataset/Contrast/low', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoint/LLE/', type=str)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_checkpoint', default=f'checkpoint/LIE/epoch{start_epoch}.pth', type=str)
    parser.add_argument('--log_path', default='log/train.txt', type=str)
    return parser


args = get_argparser().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
dataset_name = args.dataset_name
dataset_root = f'dataset/{dataset_name}/'
underexposed_path = f'dataset/{dataset_name}/underexposed'
normal_path = f'dataset/{dataset_name}/normal'
database_path = f'database/{dataset_name}'
encoding_path = f'encoding/{dataset_name}_encoding.npy'
name_path = f'encoding/{dataset_name}_name.npy'
log_root = os.path.split(args.log_path)[0]
if not os.path.exists(log_root):
    os.mkdir(log_root)
logging = log_config(args.log_path)
if not os.path.exists(args.checkpoint_root):
    os.makedirs(args.checkpoint_root)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else device}')
    logging.info("args = %s", args)

    # 设置随机数种子
    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 加载训练数据集
    train_dst = Face_dataset(root=dataset_root, phase='train', crop_size=args.crop_size)
    train_loader = data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    # 低光照图像增强模型
    enhance_net = EnhanceNet().cuda()
    # 优化器设置
    optimizer = torch.optim.Adam(enhance_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 是否从断点继续训练
    if (args.resume):
        print(f'Load model from {args.resume_checkpoint}')
        enhance_net.load_state_dict(torch.load(args.resume_checkpoint))
    else:
        enhance_net.apply(init_weights)

    # 实例化人脸识别类
    face_encodings = np.load(encoding_path)
    retinaface = Retinaface(encoding_path=encoding_path, name_path=name_path)
    # 语义分割模型
    # seg_checkpoint_path = 'checkpoint/best_deeplabv3plus_mobilenet_lapa_epoch96.pth'
    # seg_model = load_seg(seg_checkpoint_path)
    # print(f'Load seg model from {seg_checkpoint_path}')

    # 损失
    feature_cr_criterion = ContrastLoss().cuda()
    brightness_cr_criterion = ContrastBrightnessLoss().cuda()
    perceptual_criterion = PerceptualLoss()
    smooth_criterion = SmoothLoss().cuda()
    color_criterion = ColorLoss()
    # L1 = torch.nn.L1Loss().cuda()
    # color_consistency_criterion = ColorConsistencyLoss()
    seg_criterion = SegBrightnessLoss()
    face_criterion = torch.nn.CrossEntropyLoss().cuda()

    enhance_net.train()
    for epoch in range(args.start_epoch, args.epoch_num + 1):
        iteration = 0
        sum_ill = 0
        total_accuracy = 0
        interval_loss = 0
        for (images, seg_labels, face_labels) in train_loader:
            # 取对比学习正负样本
            negatives = load_samples(args.contrast_negative_path, args.batch_size, args.crop_size)
            positives = load_samples(args.contrast_positive_path, args.batch_size, args.crop_size)

            images = images.to(device, dtype=torch.float32)
            seg_labels = seg_labels.to(device, dtype=torch.long)
            face_labels = face_labels.cuda()

            enhanced_imgs, illu = enhance_net(images)

            # 对比学习损失
            contrastive_feature_loss = 0.62 * torch.mean(feature_cr_criterion(enhanced_imgs, positives, negatives))
            contrastive_brightness_loss = 0.22 * torch.mean(
                brightness_cr_criterion(enhanced_imgs, positives, negatives))

            # 特征保留损失
            perceptual_loss = torch.mean(perceptual_criterion(images, enhanced_imgs))
            color_loss = 0.05 * color_criterion(enhanced_imgs, positives)
            smooth_loss = 0.65 * smooth_criterion(images, illu)
            # color_consistent_loss = 10 * torch.mean(color_consistency_criterion(enhanced_imgs))

            # 语义分割损失
            semantic_brightness_loss = torch.mean(seg_criterion(enhanced_imgs, seg_labels))
            # origin_outputs = seg_model(transform_seg_input(images))
            # enhance_outputs = seg_model(transform_seg_input(enhanced_imgs))
            # semantic_accuracy_loss = torch.mean(L1(origin_outputs, enhance_outputs))

            # 人脸识别损失
            img1, img2 = enhanced_imgs.split(1, dim=0)
            face_cos_dis1 = retinaface.detect_image(img1)
            face_cos_dis2 = retinaface.detect_image(img2)
            # 未检测到人脸的情况
            if face_cos_dis1.shape == torch.Size([128]):
                face_cos_dis1 = torch.zeros((len(face_encodings))).cuda()
            if face_cos_dis2.shape == torch.Size([128]):
                face_cos_dis2 = torch.zeros((len(face_encodings))).cuda()
            face_outputs = torch.stack([face_cos_dis1, face_cos_dis2], dim=0).cuda()
            face_loss = face_criterion(face_outputs, face_labels)

            iteration += 1
            ill = get_average_illumination(enhanced_imgs.detach().clone())
            sum_ill += ill
            aver_ill = sum_ill / iteration

            # 计算人脸识别准确率
            accuracy = torch.mean((torch.argmax(face_outputs, dim=-1) == face_labels).type(torch.FloatTensor)).item()
            total_accuracy += accuracy
            face_accuracy = total_accuracy / iteration

            contrast_loss = contrastive_brightness_loss + contrastive_feature_loss
            feature_loss = perceptual_loss + smooth_loss + color_loss
            seg_loss = semantic_brightness_loss

            total_loss = contrast_loss + feature_loss + seg_loss + face_loss
            interval_loss += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(enhance_net.parameters(), 0.1)
            optimizer.step()

            if iteration % 10 == 0:
                interval_loss = interval_loss / 10
                logging.info(f' epoch={epoch}/{args.epoch_num}  iteration={iteration}/{len(train_loader)}  '
                             f'loss={interval_loss:.6f}  accuracy={face_accuracy * 100:.6f}%  '
                             f'illumination={aver_ill:.6f}')
                interval_loss = 0.0

        logging.info('-' * 100)
        logging.info(f'   epoch: {epoch}   accuracy: {face_accuracy * 100:.6f}%   illumination: {aver_ill:.6f}')
        logging.info('-' * 100)
        torch.save(enhance_net.state_dict(), os.path.join(args.checkpoint_root, f'epoch{epoch}.pth'))


if __name__ == '__main__':
    if not os.path.exists(encoding_path) or not os.path.exists(name_path):
        if not os.path.exists(database_path):
            load_face_database(normal_path, database_path)
        face_encoding(database_path, encoding_path=encoding_path.split('.npy')[0],
                      name_path=name_path.split('.npy')[0])
    train()
