**比赛方案说明**

参赛队伍：Expelliarmus

ID：CChan

<br>

**赛题简介**

本次飞桨基线挑战赛沿用2019国际大数据竞赛赛题，要求选手基于遥感影像和互联网用户行为，使用飞桨设计一个城市区域功能的分类模型。

<br>

**整体思路**

本次比赛是沿用2019年百度点石平台已经举办过的[*Urban Region Function Classification*](https://dianshi.baidu.com/competition/30/rule) 比赛，区别在于模型框架的使用增加了限制，即选手“必须使用深度学习平台飞桨进行模型的设计、训练和预测，不允许使用其他相关平台、框架及任何飞桨中未包含的学习方法参赛。”。

因此，本次参赛的主要思路是沿用先前比赛开源代码中可以利用的部分，减少重复工作，并使用基于paddle开发的模型，替换掉先前比赛方案中不符合本次规则的模型。

具体地，本次比赛使用了[*官方基线模型*](https://aistudio.baidu.com/aistudio/projectDetail/176495)和先前比赛中top2队伍[[]{#OLE_LINK2 .anchor}]{#OLE_LINK1 .anchor}海疯习习在GitHub上开源的[*特征提取代码*](https://github.com/zhuqunxi/Urban-Region-Function-Classification)，并结合自己使用PaddlePadlle搭的MLP模型对上述提取的特征进行训练。

参考链接：

\[1\] 百度点石平台：[*https://dianshi.baidu.com/competition/30/rule*](https://dianshi.baidu.com/competition/30/rule)

\[2\] 官方基线模型：[*https://aistudio.baidu.com/aistudio/projectDetail/176495*](https://aistudio.baidu.com/aistudio/projectDetail/176495)

\[3\] 海疯习习队伍开源代码：[*https://github.com/zhuqunxi/Urban-Region-Function-Classification*](https://github.com/zhuqunxi/Urban-Region-Function-Classification)


<br>


**比赛方案贡献**

由于本次参赛使用了先前top选手的特征提取代码，而这份代码其实对比赛结果提供了极大的贡献，因此这里先说明一下属于自己做的工作，以区分自己比赛方案的贡献：

(1)  基于PaddlePaddle框架搭建了MLP模型，并封装了MLPClassifier，提供了fit()、predict\_prob()、score()、save\_model()、load\_model()接口，方便模型训练预测调用。具体参见代码中的models.py文件。

(2)  对官方基线模型进行如下修改：

​	a.  修改npy生成文件代码，使用multiprocessing多进程处理，加快处理速度；

​	b.  修改reader函数和infer函数，使其可以batch预测，加快预测速度；

​	c.  添加了k折交叉验证代码，及stacking方式生成基线模型特征代码

(3)  使用MLP模型进行特征筛选，具体做法是：

​	a.  划分训练验证集，并使用全部特征训练MLP模型；

​	b.  按顺序依次shuffle验证集的每一列特征，并在前面训练的模型上进行预测，如果预测分数不变或者升高了，	说明这一列特征其实不起作用，可以将该特征剔除。具体参见代码中的train\_select.py文件。

(4)  后期使用bagging方式训练多个模型，即每次训练前都对样本和特征进行采样，保证模型训练结果的多样性，提高模型融合效果。

<br>

**比赛思路**

沿用top队伍海疯习习的比赛思路，具体参考大佬的博客：

[*https://www.cnblogs.com/skykill/p/11273640.html*](https://www.cnblogs.com/skykill/p/11273640.html)

(1)  使用官方基线模型提取特征，具体代码参见文件夹train\_multimodel

(2)  使用海疯习习队伍开源代码提取特征，其中包括三类特征：

a.  basic 特征

> ​	“给定一个地区的访问数据，我们提取该地区不同时间段的统计特征（包括 sum, mean, std, max, min, 分	位数25，50， 75这8个统计量）。不区分用户的特征：24小时，24小时相邻小时人数比值，节假日，工作	日，休息日，等等。区分用户的特征：1） 一天中，最早几点出现，最晚几点出现，最晚 减去 最早， 一天	中相邻的最大间隔小时数。 2）沿着天数的，每个小时的统计特征。 等等”

—— 引用自大佬博客。

b.  local 特征

> “用户的时间轴上的天数，小时数，一天中最早出现和最晚消失的时间以及其时间差，一天中相邻时间的最大间隔小时数；以及节假日的相应特征（由于内存限制，我们对于节假日的特征，只提取了部分特征，天数，小时数）, 这边我们节假日分的稍微粗糙点。”
>

—— 引用自大佬博客。

a.  global特征

在提取local特征的方法下，使用部分basic特征替换掉local特征变量，具体方法参见大佬博客

1.  使用前文提到的特征筛选方法从basic特征中筛选部分特征。

2.  在提取global特征前，继续从basic特征中筛选出50个特征，用于构造global特征。

3.  用官方基线模型特征和大佬的三类特征一起训练MLP模型，使用4折交叉验证，最终得分为0.885+

4.  使用前文提到的bagging训练方法，训练50个MLP模型进行融合，最终得分为0.887+。

注：以上MLP模型层设置均为（256，128，64）。

<br>

**代码目录及说明**

code

 ├─data：数据存放目录

 │ ├─test\_image：测试图片

 │ ├─test\_visit：测试文本

 │ ├─train\_image：训练图片

 │ └─train\_visit：训练文本

 └─work

 ├─data\_processing：数据预处理

 │ ├─get\_basic\_file：记录训练测试文件及训练标签

 │ └─get\_npy：生成npy文件

 ├─feature\_extracting：特征提取及筛选

 │ ├─Basic\_feature：basic特征

 │ │ ├─Code\_Basic\_feature\_1

 │ │ └─Code\_Basic\_feature\_2

 │ ├─UserID\_feature\_global：global特征

 │ └─UserID\_feature\_local：local特征

 ├─train\_all：使用4折交叉训练模型（score：0.885）

 ├─train\_bagging：使用bagging的方式训练模型（score：0.887）

 └─train\_multimodel：官方基线模型特征



注：属于已有开源代码的包括：

A.  修改自官方基线模型：

work\\data\_processing\\get\_npy\\get\_npy.py

work\\train\_multimodel\\multimodel.py

work\\train\_multimodel\\train\_utils.py

A.  来自GitHub开源代码：

（网址：[*https://github.com/zhuqunxi/Urban-Region-Function-Classification*](https://github.com/zhuqunxi/Urban-Region-Function-Classification)）

work\\data\_processing\\get\_basic\_file\\\*\*

work\\feature\_extracting\\Basic\_feature\\Code\_Basic\_feature\_1\\Config.py

work\\feature\_extracting\\Basic\_feature\\Code\_Basic\_feature\_1\\feature.py

work\\feature\_extracting\\Basic\_feature\\Code\_Basic\_feature\_1\\main.py

work\\feature\_extracting\\Basic\_feature\\Code\_Basic\_feature\_2\\Config.py

work\\feature\_extracting\\Basic\_feature\\Code\_Basic\_feature\_2\\feature.py

work\\feature\_extracting\\Basic\_feature\\Code\_Basic\_feature\_2\\main.py

work\\feature\_extracting\\UserID\_feature\_global\\Config.py

work\\feature\_extracting\\UserID\_feature\_global\\function\_global\_feature.py

work\\feature\_extracting\\UserID\_feature\_global\\function.py

work\\feature\_extracting\\UserID\_feature\_global\\main.py

work\\feature\_extracting\\UserID\_feature\_local\\\*\*

<br>

**代码运行顺序**

1. 进入data\_processing/get\_basic\_file

   (1) python get\_label.py: 生成训练标签

   (2) python get\_train\_test\_csv.py：记录训练visit文件（csv）

   (3) python get\_train\_test\_txt.py：记录训练visit、测试image文件（txt）

2. 进入data\_processing/get\_basic\_file

   (1) python get\_npy.py: 生成官方基线用到的npy数组

3. 进入work\\feature\_extracting\\Basic\_feature\\Code\_Basic\_feature\_1

   (1) python main.py: 生成第一组basic特征

   (2) python merge20.py: 将该组一半的basic特征合并，用于特征筛选

   (3) python train\_select.py: 利用MLP筛选特征，生成select\_index.npy

4. 进入work\\feature\_extracting\\Basic\_feature\\Code\_Basic\_feature\_2

   (1) python main.py: 生成第一组basic特征

   (2) python merge20.py: 将该组一半的basic特征合并，用于特征筛选

   (3) python train\_select.py: 利用MLP筛选特征，生成select\_index.npy

5. 进入work\\feature\_extracting\\Basic\_feature

   (1) python train\_select.py: 利用MLP筛选前面两组特征

   (2) python merge.py: 合并筛选后的特征，生成最终的basic特征

6. 进入work\\feature\_extracting\\UserID\_feature\_local

   （依次运行生成八组local特征）

   (1) python normal\_local.py

   (2) python normal\_hour\_local.py

   (3) python normal\_hour\_local\_std.py

   (4) python normal\_work\_rest\_fangjia\_hour\_local.py

   (5) python normal\_work\_rest\_fangjia\_hour\_local\_std.py

   (6) python normal\_work\_rest\_fangjia\_local.py

   (7) python data\_precessing\_user\_id\_number\_holiday.py

   (8) python data\_precessing\_user\_id\_number\_hour.py

7. 进入work\\feature\_extracting\\UserID\_feature\_global

   (1) python train\_select.py: 在basic特征上继续筛选出50个特征

   (2) python user\_place\_visit\_num.py: 用户访问地点计数

   (3) python main.py: 利用筛选的50个特征生成global特征

   (4) python merge.py: 合并，得到最终的global特征

8. 进入work\\train\_multimodel

   (1) sh download\_pretrain.sh: 下载SE\_ResNeXt50预训练模型

   (2) python train.py：k折交叉训练官方基线模型，预测概率值作为特征

9. 进入work\\train\_all

   (1) python train4fold.py: 利用MLP模型和前面生成的所有特征，四折交叉训练，预测结果线上得分为：0.885+

10. 进入work\\train\_bagging

    (1) python train.py: 利用bagging的策略训练50个MLP模型

    (2) python infer.py: 利用前46个模型预测测试集，概率值平均求和，结果线上得分为：0.887+


