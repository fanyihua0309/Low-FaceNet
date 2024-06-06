import os

data_path1 = 'path to gt images'
data_path2 = 'path to output images'

os.system(f'python lpips_2dirs.py -d0 {data_path1} -d1 {data_path2} --use_gpu')
