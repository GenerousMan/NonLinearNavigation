import numpy
from sklearn import metrics, datasets, manifold
from scipy import optimize
from matplotlib import pyplot
import pandas
import collections

def calculate_distance(x, y):
    d = numpy.sqrt(numpy.sum((x - y) ** 2))
    return d

# 计算矩阵各行之间的欧式距离；x矩阵的第i行与y矩阵的第0-j行继续欧式距离计算，构成新矩阵第i行[i0、i1...ij]
def calculate_distance_matrix(x, y):
    d = metrics.pairwise_distances(x, y)
    return d


def cal_B(D):
    (n1, n2) = D.shape
    DD = numpy.square(D)                    # 矩阵D 所有元素平方
    Di = numpy.sum(DD, axis=1) / n1         # 计算dist(i.)^2
    Dj = numpy.sum(DD, axis=0) / n1         # 计算dist(.j)^2
    Dij = numpy.sum(DD) / (n1 ** 2)         # 计算dist(ij)^2
    B = numpy.zeros((n1, n1))
    for i in range(n1):
        for j in range(n2):
            B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)   # 计算b(ij)
    return B


def MDS(data, n=2):
    D = calculate_distance_matrix(data, data)
    print(D)
    B = cal_B(D)
    Be, Bv = numpy.linalg.eigh(B)             # Be矩阵B的特征值，Bv归一化的特征向量
    # print numpy.sum(B-numpy.dot(numpy.dot(Bv,numpy.diag(Be)),Bv.T))
    Be_sort = numpy.argsort(-Be)
    Be = Be[Be_sort]                          # 特征值从大到小排序
    Bv = Bv[:, Be_sort]                       # 归一化特征向量
    Bez = numpy.diag(Be[0:n])                 # 前n个特征值对角矩阵
    # print Bez
    Bvz = Bv[:, 0:n]                          # 前n个归一化特征向量
    Z = numpy.dot(numpy.sqrt(Bez), Bvz.T).T
    # print(Z)
    return Z
