import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

DEBUG=True

def debug(*args ,**kwargs):
    global DEGUG
    if DEBUG:
        print(*args,**kwargs)

def phi(Y,mu_k,cov_k):  # 第 k 个模型的高斯分布密度函数
    norm=multivariate_normal(mean=mu_k,cov=cov_k)
    #return nor.pdf()

    return np.mat(norm.pdf(Y)).T

def expectation(Y,mu,cov,alpha): #EM算法中的E-step ，计算每个模型对样本的响应度
    #number of sample
    N=Y.shape[0]
    #number of K
    K=alpha.shape[0]
    assert N>1,"there must be more than one sample"# 要求样本数和模型个数必须大于1
    assert K>1,"there must be more than one model"
    gamma=np.mat(np.zeros((N,K)))
    prob=np.mat(np.zeros((N,K)))

    for k in range(K):
        a=phi(Y, mu[k], cov[k])
        print(a.shape)
        print(prob[:,k].shape)
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    for i in range(K):
        gamma[:,i]=alpha[i]*prob[:,i]
    for j in range(N):
        gamma[j,:]=gamma[j,:]/np.sum(gamma[j,:])
    return gamma


def maximize(Y,gamma):
    #num of sample ,num of feature
    N,D=Y.shape
    #num of model  gamma N*K
    K=gamma.shape[1]
    mu=np.zeros((K,D))  # mu 为均值多维数组，每行表示一个样本各个特征的均值
    cov=[]      # cov 为协方差矩阵的数组
    alpha=np.zeros(K)
    for i in range(K):
        NK=np.sum(gamma[:,i])
        for j in range(D):
            mu[i,j]=np.sum(np.multiply(gamma[:,i],Y[:,j]))/NK
        cov_k=np.mat(np.zeros((D,D)))
        for s in range(N):
            cov_k+=gamma[s,i]*(Y[s]-mu[i]).T*(Y[s]-mu[i])/NK
        cov.append(cov_k)
        alpha[i]=NK/N
    cov=np.array(cov)
    return mu,cov,alpha

def scale_data(Y):   # 将所有数据都缩放到 0 和 1 之间
    for i in range(Y.shape[1]):
        max_=Y[:,i].max()
        min_=Y[:,i].min()
        Y[:,i] = (Y[:,i] -min_)/(max_-min_)
    debug("Data scaled")
    return Y

def init_params(shape,K):  # 初始化模型参数
    N,D=shape
    mu=np.random.rand(K,D)
    cov=np.array([np.eye(D)]*K)
    alpha=np.array([1.0/K]*K)
    debug("parameters initialized")
    debug("mu:",mu ,"cov:","alpha:",alpha,sep="\n")
    return mu,cov,alpha

def GMM_EM(Y,K,times):  # 高斯混合模型 EM 算法  times是EM算法迭代的次数
    Y=scale_data(Y)
    mu,cov,alpha=init_params(Y.shape,K)
    for i in range(times):
        gamma=expectation(Y,mu,cov,alpha)
        mu,cov,alpha=maximize(Y,gamma)
    debug("{sep} Result{sep}".format(sep="-"*20))
    debug("mu:", mu, "cov:", "alpha:", alpha, sep="\n")
    return mu,cov,alpha





