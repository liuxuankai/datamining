#kmeans
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import  math

def loaddataset(filename): #加载数据
    fr=open(filename)
    mat=[]
    for line in fr.readlines():
        s1=line.strip().split()
        s2=list(map(float,s1))
        mat.append(s2)
    return mat

#def distance(vecA, vecB):
  #  return sqrt(sum((vecA-vecB)**2))


def choosecenter(dataset,k):  #选择每个类的中心点
    col=dataset.shape[1]
    center=mat(zeros((k,col)))
    for i in range(col):
        min1=min(dataset[:,i])
        min1=min1.astype('float')
        max1=max(dataset[:,i])
        max1=max1.astype('float')
        length=float(max1-min1)
        center[:,i]=min1+length*random.rand(k,1)
    return center


def kmeans666(dataset,k):
    row=dataset.shape[0]
    col=dataset.shape[1]
    center=choosecenter(dataset,k)
    centernum=mat(zeros((row,1)))
    taglabel=mat(zeros((row,1)))      #every point label
    state=True
    while state:  #判断是否继续循环，当每一个cluster的中心点不再改变时，停止循环
        state=False
        for i in range(row):
            minlen=inf
            tag=-1
            for j in range(k):

               # print(dataset[i,:],center[j,:])
                dis=sqrt(sum(power(dataset[i,:] - center[j,:],2)))
                #dis=distance(dataset[i,:],center[j,:])
                if dis<minlen:
                    minlen=dis
                    tag=j
            taglabel[i]=tag
            if centernum[i]!=tag:
                state=True
            centernum[i]=tag
        for s in range(k):
            temp=dataset[nonzero(centernum[:]==s)[0]]
            center[s,:]=mean(temp,axis=0)
    return row,center,centernum,taglabel




datmat=mat(loaddataset('kmeanstestset.txt'))

x,y= np.loadtxt('kmeanstestset.txt', delimiter=None, unpack=True)
z=np.loadtxt("kmeanstestset.txt")
plt.scatter(x,y,c='green',marker='+')

row,centroid,centernum,taglabel=kmeans666(datmat,2)
#plt.scatter([centroid[:,0]],[centroid[:,1]],color='r')
#plt.scatter([centroid[:,0]],[centroid[:,1]],c='r')



class0=np.array([z[i] for i in range(row) if taglabel[i]==0])
class1=np.array([z[i] for i in range(row) if taglabel[i]==1])
#class2=np.array([z[i] for i in range(row) if taglabel[i]==2])
#class3=np.array([z[i] for i in range(row) if taglabel[i]==3])



plt.plot(class0[:,0],class0[:,1],'rs',label="class0")
plt.plot(class1[:,0],class1[:,1],'bo',label="class1")
#plt.plot(class2[:,0],class2[:,1],'gs',label="class2")
#plt.plot(class3[:,0],class3[:,1],'ko',label="class3")

plt.legend(loc="best")
plt.title("Kmeans")
plt.show()









