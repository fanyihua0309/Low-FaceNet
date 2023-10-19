import os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms
from metric.psnr import compute_psnr
from metric.ssim import compute_ssim
from metric.unique.unique import unique_score
from utils.general import AverageMeter


def load_image(img_path):
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert('RGB')
    img = trans(img).unsqueeze(0)
    img = img.cuda()
    return img

def full_referenced_assessment(target_path, gt_path):
    psnr = AverageMeter()
    ssim = AverageMeter()
    with tqdm(total=len(os.listdir(target_path)), desc=f'Full-referenced') as pbar:
        for index, target_img_name in enumerate(os.listdir(target_path)):
            target_img_path = os.path.join(target_path, target_img_name)
            gt_img_path = target_img_path.replace(target_path, gt_path)
            target_img = load_image(target_img_path)
            gt_img = load_image(gt_img_path)
            cur_psnr = compute_psnr(target_img, gt_img)
            cur_ssim = compute_ssim(target_img, gt_img)
            psnr.update(cur_psnr)
            ssim.update(cur_ssim)
            pbar.set_postfix(**{
                'PSNR': f'{psnr.avg: .6f}',
                'SSIM': f'{ssim.avg: .6f}',
            })
            pbar.update(1)
    print(f'PSNR: {psnr.avg}')
    print(f'SSIM: {ssim.avg}')


def no_referenced_assessment(path):
    unique = AverageMeter()
    with tqdm(total=len(os.listdir(path)), desc=f'No-referenced') as pbar:
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            cur_unique = unique_score(img_path)
            unique.update(cur_unique)
            pbar.set_postfix(**{
                'UNIQUE': f'{unique.avg: .6f}',
            })
            pbar.update(1)
    print(f'UNIQUE: {unique.avg}')
