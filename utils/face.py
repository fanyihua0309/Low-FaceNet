import os
import shutil
from PIL import Image
from tqdm import tqdm
from face_module.retinaface import Retinaface
from utils.general import get_mode


def load_face_database(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    mode = get_mode(from_path)
    if mode == 'isfile':
        face_dict = {}
        for img_name in os.listdir(from_path):
            name = img_name.rsplit('_', 1)[0]
            if name not in face_dict.keys():
                face_dict[name] = [img_name]
            else:
                face_dict[name].append(img_name)
        multiple_images_group = 0
        for name in face_dict.keys():
            if len(face_dict[name]) > 1:
                multiple_images_group += 1
        print(f'Total {len(face_dict)} group face images. {multiple_images_group} group with more than one image.')
        with tqdm(total=len(os.listdir(from_path)), desc=f'Load face database', postfix=dict,
                  mininterval=0.3) as pbar:
            for name in face_dict.keys():
                for index, img_name in enumerate(face_dict[name]):
                    if index == 0:
                        from_img_path = os.path.join(from_path, img_name)
                        img = Image.open(from_img_path)
                        to_img_path = os.path.join(to_path, img_name)
                        img.save(to_img_path)
                    pbar.update(1)
    elif mode == 'isdir':
        with tqdm(total=len(os.listdir(from_path)), desc=f'Load face database', postfix=dict, mininterval=0.3) as pbar:
            for identity in os.listdir(from_path):
                dir_path = os.path.join(from_path, identity)
                for index, img_name in enumerate(os.listdir(dir_path)):
                    if (index == 0):
                        img_path = os.path.join(dir_path, img_name)
                        img = Image.open(img_path)
                        img.save(os.path.join(to_path, img_name))
                pbar.update(1)


# path: 人脸数据库所在的路径
def face_encoding(path, facenet_model_path=None, encoding_path=None, name_path=None):
    print('Face encoding')
    retinaface = Retinaface(1, facenet_model_path=facenet_model_path, encoding_path=name_path, face_names_path=name_path)
    image_paths = []
    names = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image_paths.append(img_path)
        parsed_name = img_name.rsplit('_', 1)[0]
        names.append(parsed_name)
    retinaface.encode_face_dataset(image_paths, names, encoding_path, name_path)