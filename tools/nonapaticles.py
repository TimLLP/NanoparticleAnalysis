import matplotlib.pyplot as plt
import locale
plt.rcParams['font.sans-serif']= ['SimHei']
plt.rcParams['axes.unicode_minus']= False

small = 2561 + 80593 + 94710
medium = 1729 + 7660 + 56654
large = 56 + 216  + 1172
data = [large, medium, small]
# data = [1172, 56654, 94710]
label = ['大颗粒', '中颗粒', '小颗粒']
plt.pie(data, labels = label, autopct='%.2f%%')
plt.title('不同大小纳米颗粒在总的纳米颗粒中百分比')
plt.show()

# /home/swu/anaconda3/envs/seg/lib/python3.7/site-packages/matplotlib/mpl-data/matplotlibrc