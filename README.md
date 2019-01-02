# LFW数据库使用指南

[TOC]

## 数据形式^1^

首先访问[LFW](http://vis-www.cs.umass.edu/lfw/)的网站，首页有四个部分需要我们留意：

### Explore the database

这部分是对数据库的一个展示。

### Download the database

这里是我们可以公开下载的数据，这里我们主要关注四个版本的数据库：

####  All images as gzipped tar file

这里是最原始的数据库版本，总共包含13233张人脸图片，5749个人，其中1680个人包含两站及以上图片。

文件的组织形式为以各个不同的人名命名的文件下，包含该人名所具有的图片，例如：

> AJ_Cook_0001.jpg

命名格式为姓名_第几张图片，格式为JPEG。

####  All images aligned with deep funneling

利用Deep funneling技术对原始数据库人脸进行矫正的版本。

####  All images aligned with funneling

利用funneling技术对原始数据库人脸进行矫正的版本。

####  All images aligned with commercial face alignment software

利用一些商业软件对原始数据库经过矫正的版本。

LFW数据库图片的获取是通过使用 ` Viola-Jones face detector` 进行人脸检测获取的结果（即我们通常使用的 ` Harr + Adaboost` 级联检测器），然后将结果在各个维度上扩大2.2倍进行裁剪到$250 \times 250$尺寸（超出的部分用像素值为0进行填充），将不符合要求的检测结果手动删除。这样做的目的可以使我们可以直接利用检测的结果，专注于我们识别系统的搭建，并且可以直接将我们设计的识别系统与检测系统进行结合使用。

### Training, Validation, and Testing

#### 问题定义

这部分内容说明如何利用LFW数据库进行算法的训练和测试。首先需要明确的是，LFW数据库是用于检验人脸验证问题(Face Verification)的数据库，即给定一对图片，判断这一对图片是否是同一人。

在进行训练集和测试集的划分时，需要注意的是：

> _A critical aspect of LFW database is that for any given training-testing split, the people in each subset are mutually exclusive._

这与人脸识别(Face Identification)问题中，针对测试图片使用诸如最近邻匹配寻找训练图片中的人进行识别的方式是有本质区别的。LFW所采用的方式旨在区分之前没有看过图片对中任何个体的情况。另一方面，LFW数据库的人脸验证可以认为是` Learning from One Example`的一种特殊情况，但是两者存在着以下两点区别：

1. 在` Learning from One Example(Per person)`情况中，我们在训练期间可以利用图片对中的训练图片，而在LFW的`Unseen Pair match`问题中，只有在测试的时候才能利用图片对中训练图片；
2. 另一个重要区别即是我们之前提及的，在` Learning from One Example(Per person)`情况中，测试图片会在训练期间采用的图片中寻找一个最佳匹配，而LFW的`Unseen Pair match`情况下，只是判断一对图片是否为同一人，任何两对图片判断都是互相独立的，不借助于其他图片对的判断。

#### 使用范例

在我们进行算法评价时，使用正确的训练测试方式进行合理的判断是至关重要的，任何不正确的使用都会导致结果的偏差而影响我们的判断。

LFW将数据组织为两部分：View1和View2。View1是用于模型选择和调整算法使用的，View2用于最终的模型评价。以下是具体介绍：

#####  View1

这部分首先我们需要厘清的一个概念就是在我们利用图片对进行训练时，拥有两种训练模式，区别仅在于我们是否能够利用已有的图片对之间的传递性，产生新的匹配或不匹配的图片对，例如：

```
我们有两对匹配的图片对：
George_W_Bush_0010	George_W_Bush_0012
以及
George_W_Bush_0042	George_W_Bush_0050
我们可以生成新的匹配对：(10,42),(10,50),(12,42),(12,50)以及结合其他人的图片生产新的不匹配图片对等等。
```

具体对应到以下两种训练模式：

- Image-Restricted Training

  该模式不允许使用人物姓名具有的传递性产生新的图片对。将View1划分为两部分数据集：

  - [`pairsDevTrain.txt`](#文件格式)文件包含用于训练的图片对，具有1100对匹配的图片对和1100对不匹配的图片对。
  - [`pairsDevTest.txt`](#文件格式)文件包含用于测试的图片对，具有500对匹配的图片对和500对不匹配的图片对。

  为了符合我们之前的问题定义，在训练集中出现的人是不会出现在测试集中的。

- Unrestricted Training

  该模式下允许利用人物姓名具有的传递性产生任意多新的图片对。将View1同样划分为：

  - [peopleDevTrain.txt](#文件格式) 用于产生训练需要的任意数量的图片对。
  - [peopleDevTest.txt](#文件格式) 用于产生测试需要的任意数量的图片对。

View1数据集使用的目的是帮助使用者针对自己使用的算法确定一个最优的模型，因此使用者可以多次地利用训练集训练并在测试集上测试，以调整自己的模型，而不用担心因此会评价结果造成偏差，因为我们最终的评价都是依据View2数据集来进行的。

#####  View2

View2数据集用于最终对我们的模型进行评价，这部分数据集的使用也是需要我们极其小心的地方。View2数据集的目的是对我们获得的__模型__进行评价，利用十折交叉验证的方式将数据集进行划分，在每一轮训练测试时，都可以利用模型得到一个__分类器__，在对应测试集上进行测试。经过十轮不同训练集上训练得到分类器，并在各自测试集上进行测试，最终将分类结果取平均，以评估我们所使用模型的性能。这一定程度上抵消了因为训练数据导致得我们对模型进行评价的偏差。这同时也要求我们要认真厘清，什么是我们的模型，模型所具有的参数；什么是我们训练得到的分类器，以及在训练数据上得到的分类器的参数。

另一方面

> _It is critical for accuracy performance reporting that the final parameters of the classifier under each experiment be set using only the training data for that experiment._ 

View2测试时利用十折交叉验证的方式，将数据集划分为10部分，每一部分包含300对匹配的图片对和300对不匹配的图片，每次抽取其中一部分作为测试集，其余9部分作为训练集，然后将最终结果进行平均，划分结果记录在[`pairs.txt`](#文件格式)文件中。在`Image-Restricted Trianing`模式下，利用View1获得的模型，利用[`pairs.txt`](#文件格式)进行最终的模型评价。

需要注意的是，在View2数据集中，在` Unrestricted Training`模式下，除了可以利用View1获得我们的模型外，还可以使用[people.txt](#文件格式)文件来产生训练集进行训练，获得我们的模型。[people.txt](#文件格式)包含10个子集，每个子集包含不同的人数，用来产生图片对时我们只能从位于同一子集中的人产生。最终在进行模型评价（测试）时，不论是通过View1还是[people.txt](#文件格式)获得的模型，都统一使用[`pairs.txt`](#文件格式)文件得到测试结果，并且__测试数据是由pairs.txt文件明确指定的，不允许利用传递性产生新的测试数据。__

##### `Image-Restricted Training`使用方法

`Unrestricted Training`使用略有不同。

> 1. Algorithm development or model selection
>    1. Use View 1 of the database to train and test as many models, with as many parameter settings, as desired.
>    2. Retain model M which has best performance on test set.
> 2. Performance reporting
>    1. Use View 2 of the database
>    2. For i = 1 to 10
>       1. Form training set for experiment i by combining all subsets from View 2 except subset i.
>       2. Set parameters of model M using training set,producing classifier i.
>       3. Use subset i of View 2 as a test set.
>       4. Record results of classifier i on test set.
>    3. Use results from all 10 classifiers to compute the estimated mean classification accuracy and the standard error of the mean as described above.
>    4. Finally, make sure to report which training method(image-restricted or unrestricted) was used.

#####  评价指标

评价指标可以包括ROC曲线和Precision-Recall曲线等等，这里只介绍最基本的两种：

- Estimated mean accuracy

  $$\hat{\mu}=\frac{\sum_{i=1}^{10}p_{i}}{10}$$

  其中 $p_{i}$ 代表用测试集 $i$ 时获得的正确率。

- Standard error of the mean

  $$S_{E}=\frac{\hat{\delta}}{\sqrt{10}},\hat{\delta}=\sqrt{\frac{\sum_{i=1}^{10}(p_{i}-\hat{\mu})^{2}}{9}}$$

  ​

### Results 

这部分用于介绍在各个协议标准下，不同算法在LFW数据库上获得的结果排序。

##  文件格式

这里我们具体介绍一下上述提到文件的具体内容和格式，详细信息可以参照网站[README](http://vis-www.cs.umass.edu/lfw/README.txt)部分的介绍。

###  pairsDevTrain.txt

第一行数字1100，表示训练集部分先包括1100对匹配样本对，然后跟随1100对不匹配样本对。

其余2200行，每行表示一个样本对。匹配的样本对包含人物名称，该人物对应的第 $n_{1}$ 张图片和第 $n_{2}$ 张图片，共三列；不匹配的样本对包含第一个人物名称 $name_{1}$ 以及该人物的第 $n_{1}$ 张图片，第二个人物名称 $name_{2}$ 以及第二个人物的第 $n_{2}$ 张图片，共四列。

###  pairsDevTest.txt

第一行数字500，表示测试集部分先包括500对匹配样本对，然后跟随500对不匹配样本对。

其余1000行，格式与pairsDevTrain.txt相同。

###  peopleDevTrain.txt

第一行数字4038给出了训练集部分具有4038个人，随后的4038行给出了每个人的姓名，以及每个人具有多少张照片，共两列。

###  peopleDevTest.txt

第一行数字1711给出了测试集部分具有1711个人，随后的1711行给出了每个人的姓名，以及每个人具有多少张照片，共两列。

###  paris.txt

第一行数字10代表具有10个子集，300代表每个子集具有300个匹配样本对，300个不匹配样本对；之后的600行（先是300行匹配，然后300行不匹配）代表第一个子集，然后依次按照600行进行划分，形成剩余的9个子集。

匹配样本对和不匹配样本对的每行格式与pairsDevTrain.txt和pairsDevTest.txt中的相同。

###  people.txt

第一行数字10给出了子集个数，随后一行的数字601给出了第一个集合具有的人数，再之后的每行表示第一个集合中每个人的名称和该人具有的图片个数；之后一行数字代表第二个集合具有的人物个数，然后人物名称个数等，依次类推到所有的10个集合。

## 协议标准^3^

我们之前所描述的`image-restricted protocol`和`unrestricted protocol`测试方法，无法正确评价利用LFW之外数据进行训练的算法、无监督学习算法，因此在原先的基础上，这里扩充并分类成以下几个不同类别的测试标准：

###  Unsupervised

在无监督学习中，我们无法获得有关数据的标签信息，标签信息的统计情况，以及任何能够帮助我们生成标签的信息。

在具体的无监督学习测试中，我们应该预先准备好一个输入为图片对的标量值函数 $f(I,J)$ ，函数返回一个标量值 $d$ 表征这一对图片的距离或相似度；然后给定任何一个阈值 $\theta$ 即可产生分类的结果：如何 $f(I,J) > \theta$ ，则类别标记为“不同”，反之则标记为“相同”。通过设定不同的阈值可以产生一条ROC曲线。注意，在无监督学习中，我们不能通过计算分类器的Accuracy来确定一个具体的阈值。相应地，我们可以采用ROC曲线下的面积（AUC）来衡量分类器的Accuracy。

注意以下几点使用限制：

1. 标量值函数 $f(\centerdot,\centerdot)$的参数设置不能借助于标签信息，以及能够产生标签信息的其他信息
2. 不能通过查看训练或测试的结果来确定阈值

#### 使用方法

> 1. Define scalar-valued function $f(\centerdot,\centerdot)$ of two images.
> 2. For each pair of images in a test set, compute the value of  $f(\centerdot,\centerdot)$ .
> 3. Sort the imaes pairs according to their computed $f$ value. Let these value of $f$ , in sorted order be denoted $f_{(1)},f_{(2)},\dots,f_{(N)}$ .
> 4. Now consider a set of threshold equal to  $f_{(1)},f_{(2)},\dots$ etc. Each threshold will generate a performance number and hence an ROC curve.(Note that there should be one additional threshold that is less than all of the f values, and classiﬁes all pairs as mismatched.)

#### ROC曲线

除了我们之前提到的Accuracy指标，评价指标还包括Precision、Recall、F1Score等评价指标。这里重点介绍一下ROC曲线及曲线下面积（AUC）指标。在我们使用Accuracy评价指标时，如果遇到类间分布不平衡时，Accuracy指标会遇到[Accuracy Paradox](https://www.zhihu.com/question/30643044)问题，相对地，如果我们使用[ROC曲线](https://www.douban.com/note/284051363/)进行评价分类器时，会对这类情况鲁棒一些。在对分类器的结果进行评价时，需要针对不同的情况选择合适的评价指标，以避免我们的评价会产生某些偏差。具体ROC曲线的绘制参见上述使用方法，AUC面积值越大代表分类器效果越好。

###  Image-restricted with no outside data

这部分协议即是我们原始的`image-restricted protocol`，协议假设LFW之外的数据不能被使用，包括额外的图像、诸如人眼检测的工具、人脸对齐的算法或者在外部数据上训练得到的特征提取方法。如果我们要在这个协议下使用诸如人脸矫正和面部特征检测此类的工具，必须遵循以下几点：

- 不能使用任何LFW之外的数据
- 不能依赖于任何额外的标注信息，例如手动进行面部关键特征点的位置标注
- 只能在训练集上进行，不能使用任何测试集

换言之，这样的工具本质上必须是无监督的，而且这样无监督学习所采用的数据必须全部在LFW的训练集之中。注意，LFW-a（利用商业系统人脸矫正的版本）是不能在这一协议下使用的。

### Unrestricted with no outside data

这部分协议与上一协议唯一的区别在于，将`image-restricted`部分换为`unrestricted`。

### Image-restricted with label-free outside data

这部分协议允许符合以下要求的，LFW之外的数据和在LFW上的标注信息被使用：

- 额外的数据不能包含任何有关两幅图像是否“相同”或“不同”的信息
- 额外的数据不能包含有关个人的识别信息，因为这样能够创造出“相同”或“不同”的图片对

总而言之，任何额外的数据都可以被使用只要这些数据不产生（或包含），在LFW之外的带“相同”或“不同”标签信息的图片对。

###  Unrestricted with label-free outside data

这部分协议与上一协议唯一的区别在于，将`image-restricted`部分换为`unrestricted`。注意任何LFW之外，非LFW图片的名称是不允许被使用的。

###  Unrestricted with labeled outside data

这部分协议是限制最小的，使用的时候只要注意不要包含任何有关LFW测试集中的人物就可以了。

## 使用举例^2^

这部分简单介绍一下，在`image-restricted with no outside data`协议下，测试eigenfaces以及LBPH算法的结果。

###  Eigenfaces

1. 在View1数据集的训练集上计算得到eigenvectors
2. 在View1数据集的测试集上确定用于分类的最佳阈值
3. 在View2数据集上进行十折交叉验证时，在每一轮利用训练集计算得到eigenvectors，然后利用View1中获得的最佳阈值在测试集上进行分类测试

|               |  欧式距离  | Cosine距离 |
| :-----------: | :----: | :------: |
| mean accuracy | 58.43% |  58.95%  |

对比实验中，使用的eigenvectors数量确定为能够捕获80%方差信息。

###  LBPH

1. 在View1数据集的测试集上确定用于分类的最佳阈值
2. 在View2数据集上进行测试，每一对图片计算得到各自LBPH特征，然后利用View1中获得的最佳阈值对图片对进行分类，在整个数据集上得到测试结果

|   LBPH非等价类    | 不带权重Chi_square距离 | Cosine距离 |  欧式距离  |
| :-----------: | :--------------: | :------: | :----: |
| mean accuracy |      57.08%      |  55.62%  | 52.80% |

|    LBPH等价类    | 不带权重Chi_square距离 |
| :-----------: | :--------------: |
| mean accuracy |      56.02%      |

采用圆形LBP算子，半径为1，采样点个数为8；计算得到直方图时，将图像划分为 $8\times8$ 的子区域。 

## 参考文献

1. [LFW Technical Report 2007](http://cs.brown.edu/courses/cs143/2011/proj4/papers/lfw.pdf)
2. [LFW Technical Report 2008](https://hal.inria.fr/inria-00321923/)
3. [LFW Techinical Report 2014](https://pdfs.semanticscholar.org/2d34/82dcff69c7417c7b933f22de606a0e8e42d4.pdf)


