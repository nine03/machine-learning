import KNN
from numpy import *

# 生成数据集和类别标签
dataSet, labels = KNN.createDataSet()
# 定义一个未知类别的数据
testX = array([1.2, 1.0])
k = 3
# 调用分类函数对未知数据分类
outputLabel = KNN.kNN_Classify(testX, dataSet, labels, 3)
print("Your input is:", testX, "and classified to class: ", outputLabel)

testX = array([0.1, 0.3])
outputLabel = KNN.kNN_Classify(testX, dataSet, labels, 3)
print("Your input is:", testX, "and classified to class: ", outputLabel)