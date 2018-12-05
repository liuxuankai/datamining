import matplotlib.pyplot as plt
import  GMM
import numpy as np

DEBUG=True
#Y=np.loadtxt("GMMData.txt")
Y=np.loadtxt("kmeanstestset.txt")
matY=np.matrix(Y,copy=True)

K=4  #k为分类的数量
mu,cov,alpha=GMM.GMM_EM(matY,K,100)
N=Y.shape[0]
gamma=GMM.expectation(matY,mu,cov,alpha)
category=gamma.argmax(axis=1).flatten().tolist()[0]
class0=np.array([Y[i] for i in range(N) if category[i]==0])
class1=np.array([Y[i] for i in range(N) if category[i]==1])
class2=np.array([Y[i] for i in range(N) if category[i]==2])
class3=np.array([Y[i] for i in range(N) if category[i]==3])


plt.plot(class0[:,0],class0[:,1],'rs',label="class0")
plt.plot(class1[:,0],class1[:,1],'bo',label="class1")
plt.plot(class2[:,0],class2[:,1],'gs',label="class2")
plt.plot(class3[:,0],class3[:,1],'ko',label="class3")

plt.legend(loc="best")
plt.title("GMM")
plt.savefig("666", dpi = 600)
plt.show()