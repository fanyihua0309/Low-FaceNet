import os
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils import data
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from face_module.retinaface import Retinaface
from utils.general import get_mode


class Test_dataset(Dataset):
    def __init__(self, dataset_path, name_path, transform=None):
        self.transform = transform
        face_names = np.load(name_path)
        face_names = face_names.tolist()
        self.images = []
        self.labels = []
        mode = get_mode(dataset_path)
        if mode == 'isfile':
            for img_name in os.listdir(dataset_path):
                img_path = os.path.join(dataset_path, img_name)
                self.images.append(img_path)
                identity = img_name.rsplit('_', 1)[0]
                index = face_names.index(identity)
                self.labels.append(index)
        elif mode == 'isdir':
            for identity in os.listdir(dataset_path):
                dir_path = os.path.join(dataset_path, identity)
                for img_name in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img_name)
                    self.images.append(img_path)
                    index = face_names.index(identity)
                    self.labels.append(index)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index], img_path

    def __len__(self):
        return len(self.images)


def predict(dataset_name, enhanced_path, facenet_checkpoint, encoding_path, name_path):
    face_encodings = np.load(encoding_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = Test_dataset(enhanced_path, name_path, transform=transform)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False)
    retinaface = Retinaface(facenet_model_path=facenet_checkpoint, encoding_path=encoding_path,
                            name_path=name_path)
    total_accuracy = 0
    with tqdm(total=len(loader), desc=f'Predict', postfix=dict, mininterval=0.3) as pbar:
        for iteration, (images, labels, _) in enumerate(loader):
            outputs = []
            for img in images.split(1, dim=0):
                cos_dis = retinaface.detect_image(img)
                if cos_dis.shape == torch.Size([128]):
                    print(f'[!] No face detected')
                    cos_dis = torch.zeros((len(face_encodings))).cuda()
                outputs.append(cos_dis)
            labels = labels.cuda()
            outputs = torch.stack(outputs, dim=0).cuda()
            accuracy = torch.mean((torch.argmax(outputs, dim=-1) == labels).type(torch.FloatTensor)).item()
            total_accuracy += accuracy
            pbar.set_postfix(**{
                'accuracy': f'{total_accuracy / (iteration + 1) * 100: .6f}%',
            })
            pbar.update(1)
    print(f'{dataset_name} accuracy: {total_accuracy / (iteration + 1) * 100}')

