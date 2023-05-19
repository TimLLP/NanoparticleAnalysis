import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import cv2
import os
images=glob.glob(r'F:\llp\images\*')
image_names=[image.split('\\')[-1] for image in images]
image_names_noSuffix=[image_name.split('.')[0] for image_name in image_names]
len(images),len(image_names),len(image_names_noSuffix)
image_names[0:10],image_names_noSuffix[0:10]
masks=glob.glob(r'F:\llp\masks_renamed\*')
# mask_names=[mask.split('\\')[-1] for mask in masks]
all_masks_list=[]#存放所有的mask,二维数组，包含每一个image的所有mask
i=0
for image_name_noSuffix in image_names_noSuffix:
    # print(image_name_noSuffix)
    masks_list=[]
    for mask in masks:
        if mask.split('\\')[-1].__contains__(image_name_noSuffix):
            masks_list.append(mask)# 每次挑选出对应的mask路径
            # print(mask)
    # 处理一张图片后把mask存入总的list
    all_masks_list.append(masks_list)
    # i+=1
    # if i>10:
    #     break
image_masks_dict=dict(zip(image_names,all_masks_list))
i=0
for image_name,image_name_abspath in zip(image_names,images):
    print(image_name,image_name_abspath)
    image_masks=image_masks_dict.get(image_name)
    print(image_masks)
    # image_name是图片名字，
    if i>10:
        break
    i+=1
test_image=Image.open(r'F:\llp\images\data_1022.png')
test_mask1=Image.open(r'F:\\llp\\masks_renamed\\data_1022_1.png')
test_mask2=Image.open(r'F:\\llp\\masks_renamed\\data_1022_2.png')
test_image_np=np.asarray(test_image)
test_mask1_np=np.asarray(test_mask1)
test_mask2_np=np.asarray(test_mask2)
plt.figure(figsize=(30,30))
plt.subplot(4,3,1)
plt.imshow(test_image_np)
plt.subplot(4,3,2)
plt.imshow(test_mask1_np)
plt.subplot(4,3,3)
plt.imshow(test_mask2_np)
test_image_np.max()
np.unique(test_mask2_np)
# composed是合成后的图片,变化为三通道彩色
composed=np.empty([3,256,256])
composed[0]=test_image_np/255
composed[1]=test_image_np/255
composed[2]=test_image_np/255
plt.imshow(composed.transpose(1,2,0))
for rol in range(255):
    for col in range(255):
        if test_mask1_np[rol][col]==255:
             composed[0][rol][col]=255
plt.imshow(composed.transpose(1,2,0))
for rol in range(255):
    for col in range(255):
        if test_mask2_np[rol][col]==255:
             composed[1][rol][col]=255
plt.imshow(composed.transpose(1,2,0))
test_image=Image.open(r'F:\llp\images\data_102.png')
test_image_name=r'data_102.png'
test_image_masks=image_masks_dict.get(test_image_name)
test_image_masks
test_image_np=np.asarray(test_image)
test_image_masks_np=[np.asarray(Image.open(img)) for img in test_image_masks]
# composed是合成后的图片,变化为三通道彩色
composed=np.empty([3,256,256])
composed[0]=test_image_np/255
composed[1]=test_image_np/255
composed[2]=test_image_np/255
for mask in test_image_masks_np:
  # 设置覆盖的颜色，rgb三者加起来为1
    r=random.random()
    g=random.random()
    b=random.random()

    r=r/(r+g+b)
    g=g/(r+g+b)
    b=b/(r+g+b)


    for rol in range(255):
        for col in range(255):
            if mask[rol][col]==255:
                 composed[0][rol][col]=r
                 composed[1][rol][col]=g
                 composed[2][rol][col]=b
plt.imshow(composed.transpose(1,2,0))
composed_img=[]
for idx,(image_name,image_path) in enumerate(zip(image_names,images)):
    masks_path=image_masks_dict.get(image_name)
    masks_np=[np.asarray(Image.open(img)) for img in masks_path]
    image_np=np.asarray(Image.open(image_path))
    # composed是合成后的图片,变化为三通道彩色
    composed=np.empty([3,256,256])
    composed[0]=image_np/255
    composed[1]=image_np/255
    composed[2]=image_np/255


    for mask in masks_np:
      # 设置覆盖的颜色，rgb三者加起来为1
        r=random.random()
        g=random.random()
        b=random.random()

        r=r/(r+g+b)
        g=g/(r+g+b)
        b=b/(r+g+b)


        for rol in range(255):
            for col in range(255):
                if mask[rol][col]==255:
                     composed[0][rol][col]=r
                     composed[1][rol][col]=g
                     composed[2][rol][col]=b
    composed_img.append(composed)

    print(idx)
    if idx>6:
        break
composed_img[0].shape
len(composed_img)
for i in range(6):
    plt.figure(figsize=(30,30))
    plt.subplot(4,3,1)
    plt.imshow(composed_img[i].transpose(1,2,0))
    plt.subplot(4,3,2)
    plt.imshow(np.asarray(Image.open(images[i])),cmap='gray')
cv2.imwrite(r'F:\llp\composed\a.png',(composed*255).transpose(1,2,0))
save_path=r'F:\llp\composed'
for idx,(image_name,image_path) in enumerate(zip(image_names,images)):
    masks_path=image_masks_dict.get(image_name)
    masks_np=[np.asarray(Image.open(img)) for img in masks_path]
    image_np=np.asarray(Image.open(image_path))
    # composed是合成后的图片,变化为三通道彩色
    composed=np.empty([3,256,256])
    composed[0]=image_np/255
    composed[1]=image_np/255
    composed[2]=image_np/255


    for mask in masks_np:
      # 设置覆盖的颜色，rgb三者加起来为1
        r=random.random()
        g=random.random()
        b=random.random()

        r=r/(r+g+b)
        g=g/(r+g+b)
        b=b/(r+g+b)


        for rol in range(255):
            for col in range(255):
                if mask[rol][col]==255:
                     composed[0][rol][col]=r
                     composed[1][rol][col]=g
                     composed[2][rol][col]=b
    composed_img.append(composed)
    cv2.imwrite(os.path.join(save_path,image_name),
                (composed*255).transpose(1,2,0)
                )

    print(idx)
    # if idx>6:
    #     break
image_names[881]
composed.shape[1]==256
# 从880开始恢复
save_path=r'F:\llp\composed'
for idx,(image_name,image_path) in enumerate(zip(image_names,images)):
    print("\r{}".format(idx),end='')
    if idx<875:
        continue
    masks_path=image_masks_dict.get(image_name)
    masks_np=[np.asarray(Image.open(img)) for img in masks_path]
    image_np=np.asarray(Image.open(image_path))

    if image_np.shape[1]==256:
        # composed是合成后的图片,变化为三通道彩色
        composed=np.empty([3,256,256])
        composed[0]=image_np/255
        composed[1]=image_np/255
        composed[2]=image_np/255


        for mask in masks_np:
          # 设置覆盖的颜色，rgb三者加起来为1
            r=random.random()
            g=random.random()
            b=random.random()

            r=r/(r+g+b)
            g=g/(r+g+b)
            b=b/(r+g+b)
            for rol in range(255):
                for col in range(255):
                    if mask[rol][col]==255:
                         composed[0][rol][col]=r
                         composed[1][rol][col]=g
                         composed[2][rol][col]=b
        composed_img.append(composed)
        cv2.imwrite(os.path.join(save_path,image_name),
                    (composed*255).transpose(1,2,0)
                    )
    else:
        print('不是255 * 255')
