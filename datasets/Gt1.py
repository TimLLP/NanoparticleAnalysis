from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from PIL import Image
import os
# 透明度调整
def addTransparency(img, factor):
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img, img_blender, factor)
    img_temp = img
    return img_temp

# def tensor_to_PIL(tensor):
#     image = tensor.cpu().clone()
#     image = image.squeeze(0)
#     image = unloader(image)
#     return image
if __name__ =='__main__':
    image_path ='/home/swu/peng/mycode/datasets/overlap/images'
    mask_path ='/home/swu/peng/mycode/datasets/overlap/gtmasks'
    save_path ='/home/swu/peng/mycode/datasets/overlap/gtImages'
    # seg_image
    for root_image, _, image_names in os.walk(image_path):
        for image_name in image_names:
            image = Image.open(root_image+"/"+image_name)
            image = image.convert('RGBA')
            # r, g, b, a = image.split()
            # print(a)
            for root_mask, _, mask_names in os.walk(mask_path):
                for mask_name in mask_names:
                    print(root_mask + "/" + mask_name)
                    if image_name == mask_name:
                        # print.1

                        seg_image = Image.open(root_mask +"/" + mask_name)
                        seg_image = seg_image.convert('RGBA')
                        # seg_image = addTransparency(seg_image, 0.1)
                        r, g, b, a = seg_image.split()

                        # print(a)
                        # print(seg_image)
                        image.paste(seg_image, (0,0), a)  # 叠图
                        # # img.show()
                        # scale = Image.open(scale)
                        # scale = scale.resize((256, 256))  # 拼图设置要和主图相接部分的尺寸对上
                        # scale.show()
                        img = np.array(image)
                        # im = np.array(scale)  # 转化为ndarray对象
                        # im2 = np.concatenate((img1, im), axis=1)  # 横向拼接
                        # 生成图片
                        # img2 = Image.fromarray(im2)

                        # img = cv2.addWeighted(image, alpha, seg_image, meta, gamma)
                        # img = cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=seg_image)
                        # img = Image.composite(image, seg_image, a)
                        # img = cv2.bitwise_and(image,seg_image)
                        # print(img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        path = '/home/swu/peng/mycode/datasets/overlap/gtImages/{}'.format(image_name)
                        cv2.imwrite(path, img)

