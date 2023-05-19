from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import skimage.io as io
import json
import numpy as np
import random

'''
路径参数
'''
# 原coco数据集的路径
dataDir = '/home/swu/peng/data'
# 用于保存新生成的mask数据的路径
savepath = "/home/swu/peng/mycode/datasets/train/gtmasks"

'''
数据集参数
'''
classes_names = ['SEM']

def getColors():
    image = Image.open("/home/swu/peng/mycode/datasets/image1399.png")
    img = image.convert('RGBA')
    total = []
    for i in range(256):
        for j in range(256):
            r, g, b, a = img.getpixel((i, j))
        if r not in total:
            if r / 255 != 0 or r / 255 != 1:
                total.append(r / 255)
    # print(total)
    return total
# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


# 生成mask图
def mask_generator(coco, width, height, anns_list):
    print(anns_list)
    mask_pic = np.zeros((height, width, 4))
    print(height,width)
    # 生成mask
    # colors = getColors()
    # print(colors)
    for single in anns_list:
        mask_single = coco.annToMask(single)
        # 读取对应预测的图片
        # 拿到对应像素值的rgb
        a = random.random()
        b = random.random()
        c = random.random()
        for i in range(0,height):
            for j in range(0,width) :
                if mask_single[i][j] > 0:
                    mask_pic[i][j][0] = a
                    mask_pic[i][j][1] = b
                    mask_pic[i][j][2] = c
                    mask_pic[i][j][3] = 1

    return mask_pic
def mask_generator_color(masks, width, height):
    # print(anns_list)
    mask_pic = np.zeros((height, width, 4))
    print(height,width)
    # 生成mask
    # colors = getColors()
    # print(colors)
    for single in anns_list:
        mask_single = coco.annToMask(single)
        # 读取对应预测的图片
        # 拿到对应像素值的rgb
        a = random.random()
        b = random.random()
        c = random.random()
        for i in range(0,height):
            for j in range(0,width) :
                if mask_single[i][j] > 0:
                    mask_pic[i][j][0] = a
                    mask_pic[i][j][1] = b
                    mask_pic[i][j][2] = c
                    mask_pic[i][j][3] = 1

    return mask_pic
# # 生成mask图
# def mask_generator(coco, width, height, anns_list):
#     mask_pic = np.zeros((height, width, 3))
#     print(height,width)
#     # 生成mask
#     # colors = getColors()
#     # print(colors)
#     for single in anns_list:
#         mask_single = coco.annToMask(single)
#         # 读取对应预测的图片
#         # 拿到对应像素值的rgb
#         a = random.random()
#         b = random.random()
#         c = random.random()
#         for i in range(0,height):
#             for j in range(0,width) :
#                 if mask_single[i][j] > 0:
#                     mask_pic[i][j][0] = a
#                     mask_pic[i][j][1] = b
#                     mask_pic[i][j][2] = c
#     return mask_pic


# 处理json数据并保存二值mask
def get_mask_data(annFile, mask_to_save):
    # 获取COCO_json的数据
    coco = COCO(annFile)
    # 拿到所有需要的图片数据的id
    classes_ids = coco.getCatIds(catNms=classes_names)
    # 取所有类别的并集的所有图片id
    # 如果想要交集，不需要循环，直接把所有类别作为参数输入，即可得到所有类别都包含的图片
    imgIds_list = []
    for idx in classes_ids:
        imgidx = coco.getImgIds(catIds=idx)
        imgIds_list += imgidx
    # 去除重复的图片
    imgIds_list = list(set(imgIds_list))

    # 一次性获取所有图像的信息
    image_info_list = coco.loadImgs(imgIds_list)

    # 对每张图片生成一个mask
    for imageinfo in image_info_list:
        # 获取对应类别的分割信息
        annIds = coco.getAnnIds(imgIds=imageinfo['id'], catIds=classes_ids, iscrowd=None)
        anns_list = coco.loadAnns(annIds)
        # print(len(anns_list))
        # 生成二值mask图
        mask_image = mask_generator(coco, imageinfo['width'], imageinfo['height'], anns_list)
        # 保存图片
        file_name = mask_to_save + '/' + imageinfo['file_name'][:-4] + '.png'
        plt.imsave(file_name, mask_image)


if __name__ == '__main__':
    # 按单个数据集进行处理

    # 用来保存最后生成的mask图像目录
    mask_to_save = savepath
    # mkr(savepath + 'masks/')
    # 生成路径
    print(mask_to_save)
    mkr(mask_to_save)
    #训练集
    trainannFile = '{}/train/annotations/sem_train.json'.format(dataDir)
    # 处理数据
    get_mask_data(trainannFile, "/home/swu/peng/mycode/datasets/train/gtmasks")
    #
    # # 验证集
    # valannFile = '{}/train/annotations/sem_val.json'.format(dataDir)
    # # 处理数据
    # get_mask_data(valannFile, "/home/swu/peng/data/val/gt")
