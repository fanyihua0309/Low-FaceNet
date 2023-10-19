import os
from collections import namedtuple
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms


class Face_dataset(data.Dataset):
    FaceClass = namedtuple('FaceClass', ['id', 'train_id', 'category', 'color'])
    classes = [
        FaceClass(0, 0., 'background', (0, 0, 0)),
        FaceClass(1, 1., 'skin', (0, 153, 255)),
        FaceClass(2, 2., 'left eyebrow', (102, 255, 153)),
        FaceClass(3, 3., 'right eyebrow', (0, 204, 153)),
        FaceClass(4, 4., 'left eye', (255, 255, 102)),
        FaceClass(5, 5., 'right eye', (255, 255, 204)),
        FaceClass(6, 6., 'nose', (255, 153, 0)),
        FaceClass(7, 7., 'upper lip', (255, 102, 255)),
        FaceClass(8, 8., 'inner mouth', (102, 0, 51)),
        FaceClass(9, 9., 'lower lip', (255, 204, 255)),
        FaceClass(10, 10., 'hair', (255, 0, 102)),
    ]

    id_to_train_id = np.array([c.train_id for c in classes])
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)

    def __init__(self, root='dataset/LaPa-Face', image='underexposed', face_names_path='encoding/LaPa-Face_name.npy', phase='test', crop_size=256):
        self.images_dir = f'{root}/{image}'
        self.targets_dir = f'{root}/seg'
        self.phase = phase
        transform_list = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.phase == 'train':
            transform_list = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((crop_size, crop_size))
            ])
        self.transform = transform_list
        self.images = []
        self.image_names = []
        self.seg_labels = []
        self.face_labels = []
        face_names = np.load(face_names_path)
        face_names = face_names.tolist()
        for image in os.listdir(self.images_dir):
            self.image_names.append(image)
            image_path = os.path.join(self.images_dir, image)
            self.images.append(image_path)
            seg_label_path = os.path.join(self.targets_dir, image.replace('jpg', 'png'))
            self.seg_labels.append(seg_label_path)
            face_name = image.rsplit('_', 1)[0]
            if face_name in face_names:
                index = face_names.index(face_name)
            else:
                print(f'[x] {face_name} not in face_names.')
            self.face_labels.append(index)

    def encode_target(self, target):
        return self.id_to_train_id[np.array(target)]

    def decode_target(self, target):
        return self.train_id_to_color[target]

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.seg_labels[index])
        if self.transform:
            image = self.transform(image)
        target = self.encode_target(target)
        if self.transform:
            target = self.transform(target)
            target = target.squeeze()
        if self.phase == 'train':
            return image, target, self.face_labels[index]
        return image, target, self.face_labels[index], self.images[index]

    def __len__(self):
        return len(self.images)


class Test_dataset(data.Dataset):
    def __init__(self, root='dataset/LaPa-Test', iscrop=True, crop_size=256):
        self.images_dir = f'{root}/underexposed'
        self.targets_dir = f'{root}/normal'
        self.images = []
        self.targets = []

        for img_name in os.listdir(self.images_dir):
            img_path = os.path.join(self.images_dir, img_name)
            target_path = os.path.join(self.targets_dir, img_name)
            self.images.append(img_path)
            self.targets.append(target_path)

        trans_list = [transforms.ToTensor()]
        if iscrop:
            trans_list.append(transforms.Resize((crop_size, crop_size)))
        self.transform = transforms.Compose(trans_list)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)
        return image, target

    def __len__(self):
        return len(self.images)