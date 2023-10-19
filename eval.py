import argparse
import os
from utils.face import load_face_database, face_encoding
from utils.predict import predict
from utils.iqa import full_referenced_assessment, no_referenced_assessment


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='CelebA-Test', type=str,
                        choices=['CASIA-Test', 'LaPa-Test', 'LFW', 'CelebA-Test', 'CelebA-Test-2', 'real'])
    parser.add_argument('--dataset_root', default='dataset', type=str)
    parser.add_argument('--facenet_checkpoint', default='checkpoint/facenet_mobilenet.pth', type=str)
    parser.add_argument('--method_name', default='Ours', type=str)
    parser.add_argument('--log_path', default='log/eval.txt', type=str)
    return parser


args = get_argparser().parse_args()
dataset_name = args.dataset_name
dataset_root = args.dataset_root
underexposed_path = f'{dataset_root}/{dataset_name}/underexposed'
normal_path = f'{dataset_root}/{dataset_name}/normal'
database_path = f'database/{dataset_name}'
encoding_path = f'encoding/{dataset_name}_encoding.npy'
name_path = f'encoding/{dataset_name}_name.npy'
enhanced_path = f'{dataset_root}/{dataset_name}/SOTAs/{args.method_name}'
facenet_checkpoint = args.facenet_checkpoint

log_root = os.path.split(args.log_path)[0]
if not os.path.exists(log_root):
    os.mkdir(log_root)
iqa_list = ['CelebA-Test', 'CelebA-Test-2', 'real']
if dataset_name in iqa_list:
    eval_mode = 'IQA'
else:
    eval_mode = 'Recognition'


def pipeline():
    print(dataset_name)
    if eval_mode == 'IQA':
        full_referenced_assessment(enhanced_path, normal_path)
        no_referenced_assessment(enhanced_path)
    elif eval_mode == 'Recognition':
        if not os.path.exists(encoding_path) or not os.path.exists(name_path):
            if not os.path.exists(database_path):
                load_face_database(normal_path, database_path)
            face_encoding(database_path, encoding_path=encoding_path.split('.npy')[0],
                          name_path=name_path.split('.npy')[0])
        predict(dataset_name, enhanced_path, facenet_checkpoint, encoding_path, name_path)


if __name__ == '__main__':
    pipeline()
