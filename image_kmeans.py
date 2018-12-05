import numpy as np
import PIL.Image as image
import kmeans
def loadData(filepath): #加载彩色图像数据
    f = open(filepath, 'rb')
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel( (i, j) )
            data.append( [x / 256.0, y / 256.0, z / 256.0] )

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
            data.append( [x[0] / 256.0] )

    f.close()
    return np.mat(data), m, n

imgData, row, col = greyData('3.jpg')

#imgData, row, col = loadData('1.jpg')
#print(imgData)
centroid,label=kmeans.kmeans666(imgData,4)
#print(imgData, row, col)

#label = KMeans(n_clusters = 4).fit_predict(imgData)
label = label.reshape( [row, col] )

print(label)
pic_new = image.new( 'L', (row, col) )
print("hhhhhhhhh")
print(pic_new)

for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (label[i,j] + 1)))

pic_new.save('result_3.jpg', 'JPEG')
