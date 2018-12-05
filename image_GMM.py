import numpy as np
import PIL.Image as image
import GMM
def loadData(filepath): #加载彩色图像数据
    f = open(filepath, 'rb')
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel( (i, j) )
            data.append( [x / 256.0, y / 256.0, z / 256.0] ) #RGB三通道

    f.close()
    return np.mat(data), m, n


def greyData(filepath): #加载灰度图像数据
    f = open(filepath, 'rb')
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x= img.getpixel( (i, j) )
           # print(x[0])
            data.append( [x[0] / 256.0] ) #只有一维的灰度值

    f.close()
    return np.mat(data), m, n

matY, row, col = loadData('1.jpg') #1.jpg是彩色图片
#matY, row, col = greyData('3.jpg')#3.jpg是灰度图片
K=3
mu,cov,alpha=GMM.GMM_EM(matY,K,20) #20代表EM算法迭代的次数

N=matY.shape[0]
gamma=GMM.expectation(matY,mu,cov,alpha)

category=gamma.argmax(axis=1).flatten().tolist()[0]
category=np.array(category)
category = category.reshape( [row, col] )

print(category)
pic_new = image.new( 'L', (row, col) )

print(pic_new)

for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (category[i,j] + 1)))

pic_new.save('GMM_result_3.jpg', 'JPEG')
