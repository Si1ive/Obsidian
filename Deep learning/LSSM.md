# 案板
VSS，AFF这种应该弱化
如果三位一体，主图应该体现三个大块，
首先冲突是三个大块的话，Unet放不下，应该把Unet简化
三个大块不是把后面的两个＋vss直接挪过来，缩小挪过来不太行，怎么简化的表示出三个SS的不同
三位一体太勉强了，SISS是分离合并，EDSS不太好叫出来，VSS更是不用说了
能不能换换角度，不同专注点？不同粒度？
轻量级 U型 适应性Mamba adpat
这样主模块图 以示意性的方式画siss和edss
siss： 扩大全局性，分离合并，更好感受不同时相 
edss：局部
感觉也不好画，并且根本没人用卡通那种图画

要解决的根本是 如果选adpat 就必须在第一张图上明显体现出哪里adpat了，现在轻量级U型倒是能体现出来了，再划分个层级，以及每层大小就能体现出来了轻量了
也就是要画出SISS怎么扩大全局性了
EDSS怎么补充局部了


淘宝图片数据存在问题
1. 包含无关格式数据，如：视频，.txt，.pdf，.ai，.psd，.eaglepack等格式
2. 图片以压缩包形式存储，部分压缩包存在解压密码
3. 存在空文件夹
4. 海报需要中文和英文内容，但存储有日文内容的
5. 图片内容是影印版非电子版，特点如：漏出纸质图片的边角
6. 图片存在水印，分辨率低不清晰等质量问题
7. 存在部分含有大量文字、黑白线稿、漫画分镜格式、人物设定集、暴力违规等内容图片


8. 用词问题
	1.  **effective不能用去掉**
	2. **Task-apdated不合适，因为暗含自适应，引导比较合适**
	3. **Mutli-Task前缀能不能加，会不会有歧义**
	4. 保持一致，比如模块名都落到名词上
	5. **Prechange不合适**
	6. **integrate 应该是integration**
	7. First order是否严谨  # **First-order Derivative operators Second-order derivative operators**
	8. Edge这么结尾不太合适，图中的Second-order Edge block也不合适，没动词 
	9. 叫multi-scale不合适，应该是level 或者attention
	10. 应该把MultiD也去掉，图中不体现
9. 双时图中体现的不明显，老师倾向于分离合并，如果起这个，那应该把线画的更明显点
10. mamba取黄色，左边就不应该是黄色

E. Other Network Details
讲一个AFF class 损失

实验怎么做
1. 先把代码改好，留不留调整的口，根据消融实验留参数接口
2. A数据描述 
	1. 取三个数据集 levir whu 
3. B实验设置 
	1. 架构细节 
	2. 训练细节（损失公式）  
	3. 评判细节（公式） 
4. C对比实验
	1. 1指标，可视化，效率
	2. 分成CNN Transformer mamba 24年三种架构一共找10个左右，先跑点根据跑的点选
	3. 三张对比数据集的图，一个数据集选4张图 结果需画图 画成红绿标记图
	4. 三张对比的指标表
	5. 对比参数量计算量表
5. D消融实验 （是要对比不同的设计，还是仅仅对比去掉这块，去掉那块的区别？）
	1. 编码器 (原始 共享权重，直接两个分支 ) 三组
	2. 解码器 双vss channel spatial (全留 去掉俩 留channel 留spatial)四组
	3. 增强 一阶 二阶 Mamba ( 原始 全无 留一阶 留二阶 都不留)五组
0 0 0 完全体

0 1 0 俩都去掉
0 2 0 留channel
0 3 0 留spatial

1 0 0 一个分支共享权重
2 0 0 两个分支
3 0 0 CNN共享分支

0 0 1 直接扣掉
0 0 2 留一阶跟后面的mamba
0 0 3 留二阶跟后面的mamba
0 0 4 只留mamba
# BUG 
1. V4版本 预测值全变成了0  
	解决：清梯度，损失函数有问题，修改了损失函数，增加了清缓存
# ST
##  版本迭代
1. 103041 dims 64 batchsize 128 v5 改了lap的融合方式，没改LN和DDB卷积，AFF激活
	1.F1 81 到第24轮之后就不涨了
2. 103140 跟10341版本一样，lr提升了10.7倍，再跑一次
	1. F1 8968 推理F1 1024 9033得分全比ChangeMamba大 比在自己电脑上跑出的效果好
	2. 学习率在倒数第二轮才发生变化，所以把轮数拉高到300再跑
3. 103144 1485 v6 batchsize 12 epoch200  
	1. 时间变得特别长 跑了18个点
	2. 得分下降了，意思AFF的激活换拉了 45轮就停了 8857
4. 103145 1577 v6 batchsize 128 lr 0.00107 
	1. F1 8921 39轮就不涨了
5. 103165 1486 v5 300轮 
	1. 仍然是8956
6. 103248 v5 1486 lr0.0008 epoch 150
7.  √103255 v6 AFF tanh lr 0.0008 epoch 150
8. 103257 v6 AFF GELU lr 0.0008 epoch 150
9. 103502 1577 v8.1 再训练一次，我觉得是训练的问题
	1. 103578再跑一次，没变
	2. 把三个张量cat到一起可能造成了信息丢失
10. 103503 1485 v6.2 的 DDB改回来了 得分下降一点
11. 103514 1486 v8.2
12. 103635 1485 v8.3 看看融合方式换成
13. 103642 1577 v9 work
14. 103797 1577 v9 把c跟out相加的去掉了 
15. 103803 1485 v10 2 2 10 显存跑满了
16. 103804 1486 v10 2 2 8 把超算显存跑满了
17. 103857 1486 v10 2 2 15 换回64 还是把显存爆了
18. 103858 1485 v10 2 2 8 64 能跑
19. 103892 1486 v10 2 2 10 64        best
20. 103962 1485 v10 2 2 12 64 爆了
21. 103963 1486 v10 2 2 14 64 爆了
22. 103964 1577 v10 4 4 12 64 爆了
23. 104021 1577 v10 2 2 12 64 把内存清理放到了每一个batch后面 掉点了
24. 104030 1486 v11 2 2 10 64  去掉增强分支中的一个分支 点猛掉 
25. 104053 1485 v12  2 2 10 64 把一阶添上去了 对于去掉分支的 猛涨
26. 104095 1577 v10 2 2 8 64 目的测试一下最后一层多了好还是少了好 把torch清理又改回来了 跑的巨慢无比  不知道什么原因 结果228掉点严重，说明最后一层是好使的
27. 104104 1486 v10 2 2 10 64  把边缘只留前两层 猛掉点 可能原因是把差异特征也给去掉了
28. 104187 1486 v10 2 2 10 64  没有加入一阶，没去掉ablap，两层边缘增强不如三层
29. 104189 1485 v12 2 2 10 64  应该是引入了ablap 掉点了
30. 105371 1485 V13 修改完了边界，vss再两个差异后 很差
31. 105380 1577 v13 vss挪到了差异融合之后 优于32
32. 105392 1486 v13 去掉了mlp 增加了相乘分支 vss再差异融合之后
33. 105619 1486 v13 把mlp找回来了，有相乘分支 
34. 105827 1486 v14 全炸了 
35. 105832 1485 v14 把边缘mlp去掉了 全炸了 
36. 105834 1577 v14 把解码器的mlp也去掉了 全炸了 
37. 105934 1577 v14 16 1E-4 把mlp全添上了 
38. 105939 1485 v14 12 1E-4
39. 105942 1486 v14 16 1E-3 37 38爆了 只有39涨点了 但也不是最高
40. 105974 1486 v14 16 1e-3  把分类器增强了看看怎么样 效果目前最好
41. 105975 1485 v15 16 0.01 还行
42. 105976 1577 v15 16 0.01 sgd bug
43. 106005 1485 v15 16 0.01 300
44. 106006 1577 v15 16 0.01 sgd 300 爆了 掉大点
45. 106080 1485 v15 conv_small kernel改成3了 0.01 200  涨点
46. 106081 1486 v15 2 2 15 掉点
47. 106083 1577 v15 2 2 10 分类器kernel 7 影响不大
48.  也就是说 kernel=3有大好处
49. 106142 1485 分类器 conv_small aff kernel都改成了3 把aff换成3 掉点  别人的结构就不动
50. 106143 1486 分类器 conv_small kernel改成了3 2 2 8  掉点
51. 106144 1577 分类器 conv_small kernel改成了3 2 4 10 掉点
52. 106220 1486 aff为1 conv_small7 1    2 2 10  暴跌
53. 106378 1485 aff1 其他均为3 1e-3 训练结果没变，测试变差很多
54. 106394 1577 v16 把编码器的合并改成非dwconv了，加了bn relu 1e-3 还行
55. 106396 1486 v16 编码器前面改成共享权重 合并部分调回原来的了 1e-3  掉点
56. 106460 1485 v16 编码器共享CNN 1e-3 装错了
57. 106476 1486 v16 把CNN 改了 合并也改了 0.01 暴跌 弄错了 把差异全去除了
58. 106477 1577 跟57一样 暴跌 但是58 57差了两个点
59. 106577 1577 v16 t300 共享CNN 移除了差异曾再跑一次 掉大点
60. 106578 1486 按45再跑 v15 0.01 效果目前就很好
61. 106579 1485 按45再跑 v15 0.01 掉点 
62. 106714 1577 v17  提交了两个一样的
63. 1577 
64. 106695 1485 v15 最好的一版 把验证集换成测试集 反而掉点了
65. 1485
66. 106716 1486 v17 val改test 弄错了
67. 106817 1485 v17 把edge里面的两个fuse改成原来的了
68. 106873 1486 v17 val改test

v15
106080 9095
106578 9044
106579 9067


编码器的 sb 改成LN linear 3*3
cat的卷积都改成1*1 非dw 加BN relu
边缘加上FADC分支 
解码器的池化改一改 卷积改成空洞的 

## 消融
1. 拿掉融合分支,也就是单独把主干拎出来
	1. 填上头
		1. 现有：对AB融合后经过核心
		2. 要对比：
			1. A,B分别进入共享核心
			2. A,B分别进入不共享核心
			都进入共享的AB后融合模块，
		
	2. BN和LN对比
	3. 修改跳接方式
		1. 直接cat
		2. 用senet
		3. 不跳接 
	要去掉的东西：
		1. 编码层：DDB
		2. 解码层：整个增强分支
1.  v1 干净的主干 和 带上头的对比

主线，执行VSS  
当AFF为None时，就是解码层第一层，主线和分支都用ABfuse,若不为None，主线用上一层的AFF结果，分支用ABfuse  
  
原本的设计中，当经过第一层解码层后，传入vssb的就是aff后的结果，不再与AB进行跳接，跳接就单独放在Laplce中，Laplace将通道数压到了1，特征信息变少了  
跳接的设计不能丢，跳接应该保留在主干网络中  
AFF,A,B如何融合，  
 1.能不能把AFF拓展到三个融合  
 2.看看其他网络跳接是怎么处理的  
     AERNET就是单纯的cat然后归一激活甚至就一轮，融合之后跟解码层核心处理完后再一个跳接  
     ConMamba不带跳接  
     changemamba中引入三个vssb处理  
  把A,B,AFFOUT，融合经过VSSB和AFFOUT经过VSSB然后跟AB融合进行对比

# Version
## Version1
1. 训练到49轮左右时lr迭代到了1e-5，得分提升了  
2. 又开了新一轮，直接把lr设定为1e-5，并且把迭代频率考察epoch数从15换成了10，得分再也提不动了  
3. 疑似10轮太少了，导致学习率拼命减，减过头更新不动了
## Version2
1. 将dim从64改成96开始  
	参数量直接从37 干到77  
	训练速度从1.5s左右干到15s左右  
	从不到十分钟干到仨小时  
1. dim改成80  
	参数量53  
	训练速度5s 一轮一小时 训练不动没训练
## Version3
1. 修改所有降通道操作，都改成先降四倍，再提两倍  
2. 移除AFF融合四个D的模块，将AFF  
3. 更改为可以融合边缘增强的架构，即解码器中每一层都输出一个结果，然后都计算到损失当中  
4. 解码器引入LSSM，边缘增强分支两个lap+一个差异融合作为分支，融合主干  
5. 修改了损失函数,使用Focal+lovasz
## Version5
1. 模型中的参数量分布不够合理
	1. 对于边缘特征信息，可以使用很少参数量的层来进行特征提取
		1. 但是对于第一个lap，我先直接把两张图片融合6通道直接压成1，毕竟还是两张图像这样一下丢失太多信息了，应该用两个lap对A,B两张图像分别特征提取，然后将两个边缘特征合起来，交给解码层
		2. 解码层，逐像素乘法，大小不一致时，必须有一个为1，因为通过expand把这个1重复，解决方案，使用repeat把lap复制到AB_VSS的大小
	2. 但是对于非浅层的特征不能使用很少参数量的层来处理，会使得特征丢失
2. 归一化使用的理由不够充分
	1. 卷积的偏置只有结合BN时才False
## Version6
修改了vmamba2中的DDB,VSSB2导致其他们版本的VSSM也都跟着变
1. 修改AFF中的tanh激活函数，tanh更适合序列任务，更替为sigmoid更适合图像的二分类任务
	1. 取值0-1，A设定为1-sigmoid，B设定为sigmoid
	2. V5的设计是主干做+1，分支做2-，修改后也保持这个思路，主干不操作，分支做1-，因为重要的还是主干，分支只是做补足
2. 修改编码层的差异提取模块，逐深度卷积改为普通卷积，然后激活从LN改为了BN，这一个改动参数量大很多
3. 调查了一下，没有把数据批数砍掉，所以不存在中间的数据批数很小，所以把除了SSM中的归一，剩下的全部改成BN试一试
4. 高斯卷积不需要偏置项

## Version7
1. 调整损失
	1. 增加解码层输出，即增加out层和损失函数
	2. 在最优的版本上加，如果加上结果更好则保留，反之剔除
	3. 观察一下参数量变化，增加out参数量变化不大，但是Flop增大很多
	4. 损失函数
		1. 二分类更适合二元交叉损失
			1. AERNet，在BCE的基础上增加了两个权重，这个权重由IOU得来，五个SWBCE堆叠得来??
## Version8
1. 先不改损失了
2. 从6.1 BEST 开始改
3. 修改解码器的跳接方式
	1. 没有AFFout时 AB融合
	2. AFFout时，三者融合
	3. 融合完都经过一个convsmall，不再区分有没有AFFout
		1. 融合模块单独拿出来写，因为融合完要给DoG提边缘
		2. SENet
## Version11
1. 