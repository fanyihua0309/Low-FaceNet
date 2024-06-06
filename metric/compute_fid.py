import os

# Reference: https://github.com/mseitzer/pytorch-fid
# Please install pytorch-fid first: pip install pytorch-fid

data_path1 = 'path to gt images'
data_path2 = 'path to output images'

os.system(f'python -m pytorch_fid --device cuda:0 "{data_path1}" "{data_path2}"')
