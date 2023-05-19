from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from PIL import Image
import os



def combine(bg,forground,alpha):
    mask_pic = np.zeros((256, 256, 3))
    for i in range(256):
        for j in range(256):
            if forground[i][j][0] > 0:
                mask_pic[i][j][0] = bg[i][j][0] * alpha + forground[i][j][0]  * (1- alpha)
                mask_pic[i][j][1] = bg[i][j][1] * alpha + forground[i][j][1] * (1- alpha)
                mask_pic[i][j][2] = bg[i][j][2] * alpha + forground[i][j][2] * (1- alpha)
            if forground[i][j][0] == 0:
                mask_pic[i][j][0] = bg[i][j][0]
                mask_pic[i][j][1] = bg[i][j][1]
                mask_pic[i][j][2] = bg[i][j][2]
    return mask_pic



if __name__ =='__main__':
    image_path = '/home/swu/peng/mycode/datasets/overlap/Img'
    mask_path = '/home/swu/peng/mycode/datasets/overlap/gtmasks'
    save_path = '/home/swu/peng/mycode/datasets/overlap/gtImages'
    # print(image_path)
    # seg_image
    for root_image, _, image_names in os.walk(image_path):
        for image_name in image_names:
            image = cv2.imread(root_image+"/"+image_name)
            for root_mask, _, mask_names in os.walk(mask_path):
                for mask_name in mask_names:
                    if image_name == mask_name:
                        seg_image = cv2.imread(root_mask +"/" + mask_name,cv2.IMREAD_UNCHANGED)
                        img = combine(image, seg_image, 0.4)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        path = '/home/swu/peng/mycode/datasets/overlap/gtImages/{}'.format(image_name)
                        cv2.imwrite(path, img)



