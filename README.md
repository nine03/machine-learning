# 邻近算法(KNN，K-NearestNeighbor K-최근접이웃 알고리즘 이론) 

### 主要原理(주요원리)
 最简单最初级的分类器是将全部的训练数据所对应的类别都记录下来，当测试对象的属性和某个训练对象的属性完全匹配时，便可以对其进行分类。但是怎么可能所有测试对象都会找到与之完全匹配的训练对象呢，其次就是存在一个测试对象同时与多个训练对象匹配，导致一个训练对象被分到了多个类的问题，基于这些问题呢，就产生了KNN。    
 KNN是通过测量不同特征值之间的距离进行分类。它的思路是：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别，其中K通常是不大于20的整数。KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。   
下面通过一个简单的例子说明一下：如下图，绿色圆要被决定赋予哪个类，是红色三角形还是蓝色四方形？如果K=3，由于红色三角形所占比例为2/3，绿色圆将被赋予红色三角形那个类，如果K=5，由于蓝色四方形比例为3/5，因此绿色圆被赋予蓝色四方形类。   
![041623504236939](https://user-images.githubusercontent.com/60682087/113731162-837bff80-9733-11eb-9cd8-99333d8a9b30.jpg)   
由此也说明了KNN算法的结果很大程度取决于K的选择。   
在KNN中，通过计算对象间距离来作为各个对象之间的非相似性指标，避免了对象之间的匹配问题，在这里距离一般使用欧氏距离或曼哈顿距离：

![041625523458191](https://user-images.githubusercontent.com/60682087/113732306-727fbe00-9734-11eb-8612-25a5a7cbe022.jpg)

接下来对KNN算法的思想总结一下：就是在训练集中数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，找到训练集中与之最为相似的前K个数据，则该测试数据对应的类别就是K个数据中出现次数最多的那个分类，其算法的描述为：

1. 计算已知类别数据集中的点与当前点之间的距离   
2. 按照距离递增次序排序   
3. 选取与当前点距离最小的k个点   
4. 确定前k个点所在类别的出现频率   
5. 返回前k个点出现频率最高的类别作为当前点的预测分类   

以下通过图来进一步解释：   
假定要对紫色的点进行分类，现有红绿蓝三个类别。此处以k为7举例，即找出到紫色距离最近的7个点。
![1619855-20190306005541234-212699402](https://user-images.githubusercontent.com/60682087/113733253-40229080-9735-11eb-85a5-6bc2da8ba927.jpg)   
分别找出到紫色距离最近的7个点后，我们将这七个点分别称为1、2、3、4、5、6、7号小球。其中红色的有1、3两个小球，绿色有2、4、5、6四个小球，蓝色有7这一个小球。
![1619855-20190306005938818-848929744](https://user-images.githubusercontent.com/60682087/113733432-6ba57b00-9735-11eb-8f84-4d5394ade8d3.jpg)
显然，绿色小球的个数最多，则紫色小球应当归为绿色小球一类。
![1619855-20190306010048163-1462847233](https://user-images.githubusercontent.com/60682087/113733618-98599280-9735-11eb-9810-095f08583717.jpg)
以下给出利用KNN进行分类任务的最基本的代码。   
KNN.py文件内定义了KNN算法的主体部分
<pre><code>
from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
    
def kNN_Classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 关于tile函数的用法
    # >>> b=[1,3,5]
    # >>> tile(b,[2,3])
    # array([[1, 3, 5, 1, 3, 5, 1, 3, 5],
    #       [1, 3, 5, 1, 3, 5, 1, 3, 5]])
    sqDiffMat = diffMat ** 2
    sqDistances = sum(sqDiffMat, axis=1)
    distances = sqDistances ** 0.5  # 算距离
    sortedDistIndicies = argsort(distances)
    # 关于argsort函数的用法
    # argsort函数返回的是数组值从小到大的索引值
    # >>> x = np.array([3, 1, 2])
    # >>> np.argsort(x)
    # array([1, 2, 0])
    classCount = {}  # 定义一个字典
    #   选择k个最近邻
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        #计算k个最近邻中各类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    #返回出现次数最多的类别标签
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex
</code></pre>
KNN_TEST.py文件中有两个样例测试。  
<pre><code>
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
</code></pre>
代码输出：
![04162552345819111](https://user-images.githubusercontent.com/60682087/113734852-a8be3d00-9736-11eb-919f-4671d043e627.JPG)
