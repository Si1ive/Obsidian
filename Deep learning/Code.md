# Code
## 环境配置
1. cv2
	pip install opencv-python   （如果只用主模块，使用这个命令安装）
	pip install opencv-contrib-python （如果需要用主模块和contrib模块，使用这个命令安装）
2. No module named 'tensorboard'
	pip install tensorboard

## 参数设计
### num_worker
1. 控制cpu的线程数，跟cpu有关，有代码测试取多少最合适
2. 跟训练和推理的效果无关，只影响速度
3. 测试代码在EGRCNN代码里面
### argparse
命令行解释器，不仅可以定义以及初始化指定参数，还可以解析用户在命令行输入的参数
1. parse = argparse.ArgumentParser()获得参数解析器对象
2. parser.add_argument添加参数
	常见的参数
	1. 参数名
		1. 不带-或--，则在命令行中按顺序输入这些参数
		2. 带-或--，则参数可输入可不输入
	2. 参数类型: str,int,float，输入时不符合要求会报错
	3. help参数: 命令行输入--help则会返回相应信息
	4. default参数: 默认
	5. action参数
		1. store_true / store_false 有这个参数，则对应赋值bool
		2. append 可多次调用同个参数，这些输入会整合到一个列表当中
		3. count 记录命令出现次数		
	6. required 用bool控制当前参数是否必选
	7. choices 用列表限定参数
	8. nargs：’?‘最多接收一个值，’*‘随意大小返回数组，’+‘至少一个值
### logger
日志输出，以及与参数的结合
#### 参数控制
1. 参数不常改动的写成json文件
2. 按行读取参数json，使用json.loads加载读入的数据，不破坏键值结构
3. 同时也可以将parse生成的键值对与其整合
#### 日志

## 数据读取
### MNIST数据集
1. torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
2. root指向.pt文件
### 普通数据集
#### 通过list读取
可以控制读哪些图片
1. img_name_list = np.loadtxt(dataset_path, dtype=np.str_)  
	if img_name_list.ndim == 2:  
	return img_name_list[:, 0]  
	return img_name_list
#### 直接读文件夹下所有文件
1. glob.glob(files + '/A' + '/*.png') 读取所有符合这个路径的文件


## 可视化
### 可视化目标
1. 损失函数

	观测损失函数来判断模型拟合情况
	训练和验证损失函数分开记录，并比较
	训练和验证损失函数都收敛，则称模型拟合
	训练时损失函数收敛，但验证损失函数不收敛，则过拟合
	训练和验证损失函数都不收敛，则欠拟合
2. 准确率：以epoch为单位
3. 学习率
4. 得分
### TensorBoard
1. writer_loss = SummaryWriter(文件夹路径,loss) 定义绘制工具实例

## 模型常见设计
### 特征逐通道相加(只要牵扯通道改变)
1. 相加之后要卷积进行融合
2. 一般使用两次卷积，第一个卷积先将通道降四倍，第二个卷积再将通道升四倍
## BUG
1. 定义不能定义成全大写，会被判定为数组？