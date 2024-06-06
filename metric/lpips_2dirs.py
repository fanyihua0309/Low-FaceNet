import argparse
import os
import lpips
import torchvision.transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0', '--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1', '--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o', '--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version)
if (opt.use_gpu):
    loss_fn.cuda()

# crawl directories
files = os.listdir(opt.dir0)
average = 0
resize = torchvision.transforms.Resize((500, 500))

with tqdm(total=len(files), desc='LPIPS') as pbar:
    for file in files:
        if (os.path.exists(os.path.join(opt.dir1, file))):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0, file)))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1, file)))

            img0 = resize(img0)
            img1 = resize(img1)

            if (opt.use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0, img1)
            average += dist01.item()
            pbar.set_postfix(**{
                'lpips': dist01.item(),
            })
            pbar.update(1)

average = average / len(files)
print(f'Average LPIPS: {average}')
