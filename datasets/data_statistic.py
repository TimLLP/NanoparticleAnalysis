#对数据集中的大小目标进行统计
import pickle, json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import itertools
import matplotlib.pyplot as plt
def countDifferentSize(gt_json,name):
    cocoGt = COCO(gt_json)
    countSmall = 0
    countMedium = 0
    countLarge = 0
    #加载注释
    for id in cocoGt.getAnnIds():
        anns = cocoGt.loadAnns(id)
        #获得ann里的元素
        ann = anns[0]
        #加载出面积(像素的数量)
        area = ann.get('area')
        if area < 32 * 32:
            countSmall = countSmall + 1
        if area >= 32 * 32 and area < 96 * 96:
            countMedium = countMedium + 1
        if area >= 96 * 96:
            countLarge = countLarge + 1
    print(name + "总粒子数{}".format(countSmall + countMedium + countLarge))
    print("小粒子个数{}".format(countSmall))
    print("中粒子个数{}".format(countMedium))
    print("大粒子个数{}".format(countLarge))
    return [countSmall, countMedium, countLarge]

def showSEM(train, test, val):
    Size = ["small", "medium", "large"]

    # 创建x轴
    xticks = np.arange(len(Size))

    fig, ax = plt.subplots(figsize=(10, 7))
    # 训练集中药物粒子统计
    ax.bar(xticks, train, width=0.2, label="train", color="red")
    # 测试集中药物粒子统计
    ax.bar(xticks + 0.2, test, width=0.2, label="test", color="blue")
    # 验证集中药物粒子统计
    ax.bar(xticks + 0.4, val, width=0.2, label="val", color="green")
    # xticks = ax.get_xticks()
    ax.set_title("Nanoparticles Statistics", fontsize=15)
    ax.set_xlabel("Nanoparticles Size")
    ax.set_ylabel("Numbers")
    ax.legend()

    # 最后调整x轴标签的位置
    ax.set_xticks(xticks + 0.2)
    ax.set_xticklabels(Size)
    addAnnotation(ax, -0.2, train, "red")
    addAnnotation(ax, 0, test, "blue")
    addAnnotation(ax, 0.2, val, "green")

    plt.show()

def addAnnotation(ax, offset, y, color):
    # print(colors)
    xticks = ax.get_xticks()
    for i in range(len(y)):
        xy = (xticks[i] + offset, y[i] * 1.01)
        s = str(y[i])
        ax.annotate(
            text=s,  # 要添加的文本
            xy=xy,  # 将文本添加到哪个位置
            fontsize=12,  # 标签大小
            color=color,  # 标签颜色
            ha="center",  # 水平对齐
            va="baseline"  # 垂直对齐
        )


if __name__ == "__main__":
    train_json = "/home/swu/peng/data/train/annotations/sem_train.json"
    test_json = "/home/swu/peng/data/test/annotations/sem_train.json"
    val_json = "/home/swu/peng/data/val/annotations/sem_val.json"
    train = countDifferentSize(train_json,"训练集")
    test = countDifferentSize(test_json,"测试集")
    val = countDifferentSize(val_json,"验证集")
    showSEM(train, test, val)