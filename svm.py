# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib import colors
from sklearn.model_selection import train_test_split

def show_accuracy(y_hat, y_test, param):
    pass

path = 'student.txt'  # 数据文件路径
data = np.loadtxt(path, dtype=float)
x, y = np.split(data, (4,), axis=1)


x = x[:, :2] #考虑身高、体重这两个特征
#x = x[:,2:4] 考虑肌肉重和基础代谢这两个特征
#x = x[:,:4] 考虑身高、体重、肌肉重和基础代谢四个特征
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
clf = svm.SVC(C=0.3, kernel='linear',decision_function_shape='ovr')
#clf = svm.SVC(C=0.8, kernel='rbf', gamma=5, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())

print('training dataset accuracy:',clf.score(x_train, y_train))  # 精度
y_hat = clf.predict(x_train)
show_accuracy(y_hat, y_train, '训练集')

print('testing dataset accuracy:',clf.score(x_test, y_test))
y_hat = clf.predict(x_test)

show_accuracy(y_hat, y_test, '测试集')

print('decision_function:\n', clf.decision_function(x_train))
print('\npredict:\n', clf.predict(x_train))



k1=0
k2=1
x1_min, x1_max = x[:, k1].min(), x[:, k1].max()  # 第0列的范围
x2_min, x2_max = x[:, k2].min(), x[:, k2].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

print('grid_test = \n', grid_test)
grid_hat = clf.predict(grid_test)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

alpha = 0.5
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
plt.scatter(x[:, k1], x[:, k2], c=np.squeeze(y), edgecolors='k', s=50, cmap=cm_dark)  # 样本
plt.plot(x[:, k1], x[:, k2], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(x_test[:, k1], x_test[:, k2], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'被测学生的肌肉重', fontsize=13)
plt.ylabel(u'被测学生的基础代谢', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'学生性别SVM二特征分类', fontsize=15)
# plt.grid()
plt.show()


