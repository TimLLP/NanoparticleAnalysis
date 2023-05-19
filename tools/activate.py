import torch
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt


def sigmod():
    x = torch.linspace(-5, 5, 200)  # 构造一段连续的数据
    x = Variable(x)  # 转换成张量
    x_np = x.data.numpy()  # plt中形式需要numpy形式，tensor形式会报错

    y_sigmoid = F.sigmoid(x).data.numpy()  # torch.nn.functional中调用sigmoid函数
    plt.plot(x_np, y_sigmoid, c='blue', label='sigmoid')
    plt.ylim((-0.2, 1.2))
    plt.legend(loc='best')
    plt.show()
def Tanh():
    x = torch.linspace(-5, 5, 200)  # 构造一段连续的数据
    x = Variable(x)  # 转换成张量
    x_np = x.data.numpy()  # plt中形式需要numpy形式，tensor形式会报错

    y_tanh = F.tanh(x).data.numpy()  # torch.nn.functional中调用tanh函数
    plt.plot(x_np, y_tanh, c='blue', label='tanh')
    plt.grid()
    plt.legend(loc='best')
    plt.show()
def relu():
    x = torch.linspace(-5, 5, 200)  # 构造一段连续的数据
    x = Variable(x)  # 转换成张量
    x_np = x.data.numpy()  # plt中形式需要numpy形式，tensor形式会报错

    y_relu = F.relu(x).data.numpy()  # torch.nn.functional中调用relu函数
    plt.plot(x_np, y_relu, c='blue', label='ReLU')
    plt.grid()
    plt.legend(loc='best')
    plt.show()
def snano():
    plt.figure(figsize=(3, 3), dpi=100)
    name = ['0.2/0.8', '0.4/0.6', '0.6/0.4', '0.8/0.2']
    ours = [49.3, 49.5, 49.6, 50.3]
    baseline = [49.1, 49.1, 49.1, 49.1]
    plt.plot(name, ours, c='blue', label='ours',marker='o',linewidth=3)
    # plt.plot(name,ours, c ='blue',alpha=0.8,marker='o',linestyle='--',linewidth=1)
    plt.plot(name, baseline, c='orange', label='single', linestyle='--',linewidth=3)
    # 绘制网格->grid
    plt.grid(alpha=1)
    plt.legend(loc='best')
    plt.xlabel('α/β')
    plt.ylabel('APs')
    plt.show()

def lnano():
    plt.figure(figsize=(3, 3), dpi=100)
    name = ['0.2/0.8', '0.4/0.6', '0.6/0.4', '0.8/0.2']
    ours = [70.2, 73.0, 72.5, 72.0]
    baseline = [72.4, 72.4, 72.4, 72.4]
    plt.plot(name, ours, c='green', label='ours', marker='o',linewidth=3)
    plt.plot(name, baseline, c='orange', label='single', linestyle='--',linewidth=3)
    # 绘制网格->grid
    plt.grid(visible=True,alpha=1)
    plt.legend(loc='best')
    plt.xlabel('α/β')
    plt.ylabel('APl')
    plt.show()

def allnano():
    plt.figure(figsize=(3, 3), dpi=100)
    name = ['0.2/0.8', '0.4/0.6', '0.6/0.4', '0.8/0.2']
    ours = [59.7, 59.8, 60.1, 60.2]
    baseline = [59.3, 59.3, 59.3, 59.3]
    plt.plot(name, ours, c='purple', label='ours', marker='o', linewidth=3)
    plt.plot(name, baseline, c='orange', label='single', linestyle='--', linewidth=3)
    # 绘制网格->grid
    plt.grid(visible=True, alpha=1)
    plt.legend(loc='best')
    plt.xlabel('α/β')
    plt.ylabel('AP')
    plt.show()
# sigmod()
# Tanh()
# relu()
snano()
lnano()
allnano()