import copy
import math
import pprint

import cv2
import numpy as np
import torch
import torchvision.transforms.functional
from PIL import Image


# ---------------------------------------------------#
#   对输入图像进行resize
# ---------------------------------------------------#
def letterbox_image_origin(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh))
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


def letterbox_image(image, size):
    _, _, ih, iw = image.shape
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    tra = torchvision.transforms.Resize((nh, nw))
    image = tra(image)
    new_image = torch.ones((image.size(0), image.size(1), h, w)).float() * (128. / 255)
    new_image[:, :, (h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


def preprocess_input(image):
    image -= np.array((104, 117, 123), np.float32)
    return image


# ---------------------------------#
#   计算人脸距离
# ---------------------------------#
def face_distance_origin(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# 计算人脸的余弦距离/余弦相似度
def cos_similarity_origin(face_encodings, face_to_compare):
    dot = np.sum(np.multiply(face_to_compare, face_encodings), axis=1)
    norm = np.linalg.norm(face_to_compare) * np.linalg.norm(face_encodings, axis=1)
    dist = dot / norm
    return dist


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return torch.empty((0))
    return torch.norm(face_encodings - face_to_compare, dim=1)


# 计算人脸的余弦距离/余弦相似度
def cos_similarity(face_encodings, face_to_compare):
    dot = torch.sum(torch.mul(face_encodings, face_to_compare), dim=1)
    norm = torch.norm(face_to_compare) * torch.norm(face_encodings, dim=1)
    dist = dot / norm
    return dist


# ---------------------------------#
#   比较人脸
# ---------------------------------#
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    cos_dis = cos_similarity(known_face_encodings, face_encoding_to_check)
    return list(dis <= tolerance), dis, cos_dis


def compare_faces_origin(known_face_encodings, face_encoding_to_check, tolerance=1):
    dis = face_distance_origin(known_face_encodings, face_encoding_to_check)
    cos_dis = cos_similarity_origin(known_face_encodings, face_encoding_to_check)
    return list(dis <= tolerance), dis, cos_dis


# -------------------------------------#
#   人脸对齐
# -------------------------------------#
def Alignment_1(img, landmark):
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]
    # 眼睛连线相对于水平线的倾斜角
    if x == 0:
        angle = 0
    else:
        angle = math.atan(y / x) * 180 / math.pi
    center = (img.shape[3] // 2, img.shape[2] // 2)
    new_img = torchvision.transforms.functional.rotate(img, angle, center=center, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    return new_img


def Alignment_1_origin(img, landmark):
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]
    # 眼睛连线相对于水平线的倾斜角
    if x == 0:
        angle = 0
    else:
        # 计算它的弧度制
        angle = math.atan(y / x) * 180 / math.pi
    center = (img.shape[1] // 2, img.shape[0] // 2)

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射函数
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


# a = torch.randn((1, 3, 4, 6))
# a.requires_grad = True
# b = torchvision.transforms.functional.rotate(a, 90)
# c = torchvision.transforms.functional.rotate(a, center=[3, 2], angle=90)
# print(c)
# print(b)
# a = torch.tensor([1,2,3])
# b = copy.deepcopy(a)
# print(a, b)
# a[1] = 555
# print(a, b)
