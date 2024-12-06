# 代码
## Vmamba
### 问题
1. VSSM里面是一个写好的编码层，我应该继承他，在他上面重写?
	应该是继承加重写
2. **直接用VSSM和Backbone_VSSM的区别是什么?**
	Backbone对VSSM里面的forward进行了重写，好像是Backbone里面根据out_indices来组织了一下输出数据
3. **model和model_ref的区别是什么**
	没看出来
4. SSM_FORWARDTYPE: "v05_noz"和v3_noz的区别是什么？
5. 怎么设计？
	关于共享的问题：共享也就是参数量少，速度快，但不灵活，更适合图像情况相似度高
	可以部分共享部分不共享，对于低级特征的提取共享权重，高级的不共享
6. 怎么实现共享不共享问题？
	得写两个layer了，前面共享的写一个，后面的不共享部分写一个
	CSSM模块因为把AB同时输入所以属于不共享部分
	有不共享，有共享，还有特殊模块，干脆不用读取layer的方式了，就拆开成共享模块，CSSM模块，不共享模块，DDM模块
7. 设置ssm_ratio和mlp_ratio的目的是什么？
8. 理顺一下哪些地方写了forward哪些地方没写
	vssm为什么也写了forward，因为Backbone_VSSM的第三方开发的吗，然后后者把前者是覆盖掉了
	makelayer中没写，里面将各个层串到block里，
	cssm模块是定义了层，所以写了forward
	cssm里面定义的ss2d没写，为什么，ss2d不也是层吗，好像是因为使用继承的forward
9. flop是什么方法?
10. initweight是什么方法？
11. 现在的问题是什么？
	1. 首先所有变化处都要考虑尺寸和维度
	2. patch嵌入层，将尺寸通过卷积向下降了两次，那么初始尺寸是不是应该调成1024和1024,这样降完之后是256，每层尺寸缩小一次，256-128-64-32
	3. 最后怎么将尺寸上升到1024呢？直接从256线性插值到1024也太奇怪了吧。MambaCD就是在最后一层上采样多采了两倍
	4. 应该先理解一下patch的目的是什么，代码看起来怎么就单纯的降尺寸扩大通道呢
	5. 不是上述原因引起的bug，还是forwardcore处出现的bug，但是昨天就从vmamba中选了个参数v05_noz，编码层就能用了，解码层现在又不行了
12. 


### 解决
1. √
2.  √
3. 应该没什么用，flo
### patch_embed
1. 两种实现方式
2. v1一层卷积一个通道优先调整一个归一
3. v2增加一个激活归一以及一个新的卷积
4. 第一层卷积都是实现补丁切分（步长和卷积核大小相同），以及通道的扩张
5. nn.LayerNorm默认对输入的数据按照从后往前对应，而数据一般为N,C,H,W,如果我们想要按通道数进行归一，那么要把C放到最后一维，并且输入一个数据
6. V2将到embed_dim分成两步卷积，作用：浅层的卷积提取的是浅层特征，浅层特征通道数弄那么多不好，所以取一半，第一步提取完再经过激活，捕捉到更多的特性后再进行第二次卷积，这样可以捕捉到更多的信息
### Vmamba层级关系
1. Backbone_VSSM ：主要写了模型里面的各个层的调用，以及权重初始化，重写VSSM的forward，调用VSSM里定义的layers
2. VSSM：1的父类，Vmamba的编码层块，将编码层的所有组件都包装到layers里面，makelayer在里面
3. make_layer：组织CSSM 和DDB
4. CSSM：组织共享部分，CSS ,缩放因子，FNN
### SS2D
1. 提供了四种实现方式
	因为MambaCD里面ssmforward取v3noz，所以我也取这个
	v3noz对应initv2
	v3noz在匹配init时都指向了initv2有什么区别呢，所以这个forwardtype到底是什么含义

### CSSM
1. 设想的创新点是，让SSM直接捕捉到A,B两张图片的全局
	矛盾点：
	1. MambaCD是在编码器提特征，解码器融合特征，而我将融合特征和提取特征都放到了编码器，解码器似乎没有工作了
	2. MambaCD的特征融合不修改扫描方式就实现了对双时相的同时特征提取，而且人家的方法更优
	3. 这似乎也不是遥感变化检测的痛点
### 可以调整的细节
1. ddb模块里面的归一层是不是有点冗余
2. decoder中的残差网络的归一层原本使batchnorm，我改成layernorm了
3. AFF模块当中，修改图像尺寸
4. 解码层mambaCD的设计是将每一层都将通道映射到最小，而不是逐层减小

### 参数编写
1. channelfirst跟归一器绑定，所以肯定要写成活的
2. 每一层都有使用，所以在最外层也就是ZZHnet处应该把自己手动改的参数都写好
### 激活函数
1. SiLU: x*sigmoid
2. GELU: x*tanh 比SiLU复杂得多

### 初始化
#### _init_weights
1. 全连接层：对模型的影响大，若随机初始化权重和偏置可能造成梯度爆炸或消失等问题，所以要将参数初始化，其中权重通过控制标准差来进行初始化，通过trunc_normal_函数实现。对于偏置置为0即可
2. 归一层：将权重全置为1，即不影响
### BUG
1. nn.sequential的forward只能接受一个输入
2. json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
	初步认定为 fowardtype设置的有问题
	vmamba和mambaCD的实现方式不同
	vmamba是有_的，编号也不一样
	mambaCD是v3_noz
	因为mambaCD是变化检测，所以想按照他的设置
	mambaCD里面forwardcore的设置也不一样
	
		目前我的代码中是vmamba的实现，所以我要改成mambacd的实现，才能跟mambacd对齐
	mambaCD获self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"
	Vmamba获得self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex
	区别就是mambaCD直接在生成forwardcore时就确定了选择SelectiveScanOflex
	可选的有三种扫描模式，分别为oflex core mamba
3. 编码层输出的差异特征尺寸大小不对，通道数是对的
	得到的d第一个尺寸为64×8×8应该是64×256×256？
	经过patchembed后尺寸就被砍为四分之一？ 不是的，是根据dim[0]来进行patch
4. 融合层拿到的d 为1 512 32 32 是对的
5. 到解码层 c1也是对的 1 1536 32 32
6. ssm.ratio是用来扩大线性层的输出特征数的，但是没看懂线性层
还是四个维度摆放顺序的错误，根本原因是channelfirst参数的设定，以及norm2d的设定的原因，标准应该是对于ss2d接受的数据和channelfirst应该匹配，比如：通道在第二维度，ss2d收到的就应该是true
# 关系盘点
1. SSM
2. S4ND引入2D的SSM
3. Mamba提出1D选择性扫描性，提出基本模型H3与gatedMLP组合
4. Vmamba引入SS2D，提出2D选择性扫描性，并且对应Mamba的Mamba块编写了VSS块
5. VisionMamba与Vmamba同级，都是据说Vmamba性能优于VisionMamba，时间也晚于Vision，
6. VSS块将mamba中的Gated MLP砍掉了，把SSM改成了四向SSM并命名为VSS
7. Vision
# Mamba
## 问题
1. mamba的时间扫描特征具体是什么？
2. CD的时间性，单单指的是双时？
3. Vmamba中的FFN是什么？
4. Vmamba中步骤D增加的MLP在哪，DWconv不是去掉了吗？

## 解决
1. A
2. A
3. 前馈模型，就俩激活一个卷积
## 引入背景
### 目标问题
#### RNN缺陷
![[Pasted image 20241012170845.png]]
1. 每一时刻的特征都是需要结合上一时刻的特征，根据这个特征来预测值，但是这个特征的容量是有限的，时间长后，早期的特征会被逐渐遗忘
2. RNN为时序，不同于CNN，不可以进行并行运算，速度慢
#### Transformer缺陷
注意力机制的二次复杂度

### 特性优势
#### 
## S4
首次在这篇论文中提出
[[[2111.00396] Efficiently Modeling Long Sequences with Structured State Spaces (arxiv.org)](https://arxiv.org/abs/2111.00396)]
视频讲解
[[MedAI #41: Efficiently Modeling Long Sequences with Structured State Spaces | Albert Gu - YouTube](https://www.youtube.com/watch?v=luCBXCErkCs)]
==S4 即ssm+Hippo+离散化==
![[Pasted image 20241014103314.png]]
### SSM

#### 状态空间
通俗理解，空间上的每一个点都蕴含信息
其中包括： 当前位置，以及当前到出口的距离，以及下一步能怎么走
其实跟一个RNN很相似，在RNN中这三个信息分别对应x（t），h（t），y（t）

但是跟RNN有本质区别，它可以输入连续序列，并预测序列
#### 推导方程
![[Pasted image 20241012173516.png]]
h'(t)中的撇可以理解成h(t+1)，是用来更新状态的，与yt方程不相干
ht存储当前状态，A存储的是历史信息，从而来更新ht
ht是ssm的核心，ht的核心则为A存储的信息
CD则存储怎么利用ht和xt来预测yt的信息
==A,B,C,D训练结束后便固定了，这些数据和一般卷积层中的权重是什么关系？==
（应该是同级的关系）
![[Pasted image 20241013152536.png]]
==对于预测y，不是直接影响的只有C和D吗？为什么图中相当于除了A都用上了?==
肯定都用上，A也用上了，不过是用的上一轮A
#### 对比RNN方程
![[Pasted image 20241012173706.png]]
1. 都是通过两个分别对应上一时刻特征和输入的矩阵来预测现在的特征，不过RNN接一个激活函数
2. RNN预测离散信息，ssm预测序列
### 特性
#### 离散化非离散化-零阶保持技术
因为要处理序列化数据，所以离散化数据要特殊处理
并且对于ssm，实际上是将连续的数据也按照离散的形式读取了
1. 收到离散的信号时，一直保留其值，直到新的离散信号出现
2. 离散信号保持的时间称作△,步长
3. 同时连续的信号也是用步长为单位来进行采样
4. 同时计数单位也不再是t，而是k，来表示第几个序列
#### 循环结构化推理
SSM的推理过程如果正常写推理，会受卷积核大小影响，推理速度不是很快
可以写成RNN那种循环形式，可以提速
#### 卷积结构化训练
目的：像CNN一样并行训练
推导算式
![[Pasted image 20241013162811.png]]
卷积核.
![[Pasted image 20241013162850.png]]
卷积过程
![[Pasted image 20241013162911.png]]
矩阵表示
![[Pasted image 20241013163152.png]]
#### 长距离依赖-Hippo
A矩阵用来记录历史状态信息
通过这个算法，可以将输入压缩
然后来找到A的最优解，从而解决长距离依赖问题

1. 在transformer中可以理解为不对信息进行压缩，只是有所权重，而上下文有窗口上限，所以总有极限
2. S4对信息进行压缩，并且有所权重，所以相对于transformer没有极限
3. 类似于拟合一个多项式近似输入
4. 评价这个多项式(measure)拟合的好坏使用EDM,这个并不是唯一的，有很多种measure方法
长距离依赖问题可归结成将上下文压缩成更小的状态，这是序列建模的基本问题
### 问题
#### 无选择性
ssm（也可以说S4）中ABCD矩阵在训练结束后不再发生改变，对于任何输入都是相同的待遇进行推理
而在Mamba中，对于不同的输入，在推理时会做有针对性的处理
当然这种针对性的动态选择也是在训练时学习的，并不是真的在推理时还会发生改变
在transformer当中注意力机制也会根据不同的token来赋值不同的权重，但是并不会压缩信息
#### 解决方案 （无选择性）
因为如果要直接把ABCD改成动态的，那么训练中就不能像CNN一样并行运算了，因为不能把ABCD直接全放卷积核里面进行运算，所以后续Mamba要针对这个问题引入新的解决方案
就是压缩信息+权重
## S6新特性
### 选择性
选择性是描述，对状态信息的策略
mamba不仅要实现全局上下文的状态存储，还要兼顾性能，也是因此称将S4升级到了S6
![[Pasted image 20241015090819.png]]
![[Pasted image 20241015091304.png]]
### 硬件感知
非卷积进行循环计算，不用CNN结构也实现了并行训练
==没有具体化扩展状态?==
### 简易架构
==将SSM架构的设计与transformer的MLP块合并为一个块(combining the design of prior SSM architectures with the MLP block of Transformers into a single block)，来简化过去的深度序列模型架构，从而得到一个包含selective state space的架构设计==

## 数据流
### SSM
![[Pasted image 20241015104258.png]]
![[Pasted image 20241015104400.png]]
xt长度为L，N=64L=10000
![[Pasted image 20241015104847.png]]
D是通道数，即一条xt有几个通道,B是批数
### Transformer
![[Pasted image 20241015105510.png]]
对每几个通道设定一个注意头
### Mamba
![[Pasted image 20241015105517.png]]
对每一个通道都设定一个SSM
# 论文
## Vmamba

### 视觉的位置敏感性
因为Mamba单向建模，并且缺乏对于图像的位置意识
选择性扫描处理文本很合适，但是不适合处理视觉
S4ND重新组织了SSM，通过外积将内核从1D扩展到了2D，SS2D将选择扫描扩展到2D
### 多向状态空间压缩模型
Mamba是单向的，即一个SSM，Vmamba为四个方向，也对应四个SSM
VisionMamba两个方向，一前一后
### 特性
1. 处理高分辨率图像时占用内存少
2. 推理速度快
3. SS2D因为分成了多个序列，对每一个序列处理一个独立的SSM所以关注上下文的时候关注的是一个序列里面的上下文，相对于注意力机制不再是关注全局的了，所以计算注意力的时间复杂度压缩到了线性，不再是平方，其实这个特性是mamba的s6带来的
4. ImageNet 分类任务，Vmamba远大于vision，对应变化检测的语义分割任务，在ADE20K上mIOU也是大于vision，所以确定魔改Vmamba
5. 不只是修改了
### 模型
1. 将图像切块，按四个方向读成四个序列，每个序列都对应放入一个S6模块进行特征提取，最后交叉合并特征



## RSCaMa
![[Pasted image 20241106111132.png]]
将Mamba引入CD，N×CaMa个组成主干，每个CaMa主要有两部分组成
### ST-SSM
增强空间变化感知
### TT-RSM
Mamba的时间扫描特征与RSICC的时间性之间存在潜在相关性
以时态交叉的方式扫描双时态特征，增强模型的时态理解和信息交互
