import argparse
from utils.enhance import enhance_images


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--underexposed_path', default='dataset/CelebA-Test/underexposed', type=str)
    parser.add_argument('--enhanced_path', default='dataset/CelebA-Test/enhanced', type=str)
    parser.add_argument('--enhance_checkpoint', default='checkpoint/Low-FaceNet.pth', type=str)
    return parser


args = get_argparser().parse_args()
# 低光照图像路径 (包含图像文件或包含子文件夹均可，无需单独处理)
underexposed_path = args.underexposed_path
# 增强图像保存路径
enhanced_path = args.enhanced_path
# 模型参数加载路径
enhance_checkpoint = args.enhance_checkpoint


if __name__ == '__main__':
    enhance_images(underexposed_path, enhanced_path, enhance_checkpoint)
