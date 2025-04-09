# 案板

本实施例提供了一种基于Mamba混合模型子任务驱动的遥感图像变化检测方法，包括如下步骤,如附图1所示：

一种基于Mamba混合模型子任务驱动的遥感图像变化检测方法，其特征在于，包括：如下步骤：

S1:将基于深度学习的遥感图像变化检测任务解耦为双时图像建模、差异特征捕捉以及目标物体重建三个子任务。

S2:搭建基于Mamba的时相交互式的编码模块，以实现双时图像的上下文依赖关系建模。

具体的，在本实施例中，所述基于Mamba的时相交互式的编码模块执行流程为：首先将两组不同相特征传入孪生的CNN残差模块提取浅层特征；再对两时相浅层特征进行层归一化，紧接经过一个初始的线性嵌入层：通过一个SiLU激活函数和深度可分离卷积层进入核心的SS2D模块，SS2D模块的输出经过层归一化；最后将其与融合前的特征残差链接，再传入非孪生的MLP模块提取深层的时相独立特征。网络结构模型如图附图2a所示。

所述CNN残差模块包括层归一层、线形层以及一个卷积层；

所述MLP模块包括层归一层以及一个扩大映射空间的线形层和一个还原映射空间的线性层；

S3:搭建带有双阶导数边缘算子和频率自适应Mamba模块的差异模块，以实现差异特征捕捉。

具体的，在本实施例中，所述带有双阶导数边缘算子和频率自适应Mamba模块的差异模块执行流程为：首先将双时图像特征传入双阶导数边缘提取模块；再将两组边缘特征分别与两时相特征残差链接，并传入差异模块；最后将具有边缘聚焦的差异信息传入频率自适应Mamba模块。网络结构模型如图附图2b所示。

其中，所述双阶导数边缘提取模块的执行流程为：首先按照符合Scharr算子初始化3*3卷积核权重，并三次旋转45°获得四个角度卷积核权重的卷积层；将特征信息传入四个卷积层获得边缘信息，然后开方、加和再开根号得到一阶导数边缘特征。通过不同权重的Guass算法做差得到的3*3卷积核权重，得到符合Laplace算子约束的卷积层；将特征信息传入上述卷积层提取二阶导数边缘特征；

其中，Guass算法如下所示：
其中，![](data:image/png;base64,R0lGODlhCwAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAAEAALAAkAhAAAAAAAAAAAZgA6ZgA6kABmtjoAADoAZjo6OjpmtjqQ22YAAGY6AGa2/5A6AJDb/7ZmALZmOraQZrb//9vb/9v///+2Zv/bkP/btv//tv//2wECAwECAwECAwECAwECAwU3ICBqEhKcASFaRkJdxyQCl6GI2XKLEFHhiwJwRzMIi42ZJZBEjhwCGUDjKFQ0EeYMgGGcBo9tCAA7)表示Guass算法权重，![](data:image/png;base64,R0lGODlhUgAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADABSABEAhQAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAADoAOjo6Ojo6Zjo6kDpmkDpmtjqQtjqQ22YAAGY6AGY6OmY6ZmZmkGZmtmaQZmaQkGaQtmaQ22a222a2/5A6AJBmAJBmOpBmZpCQOpCQZpC2kJC225Db/7ZmALZmOrZmZraQOraQZrbb/7b//9uQOtuQZtu2Ztu2kNvbttv/29v///+2Zv/bkP/btv//tv//2wECAwECAwECAwECAwECAwECAwECAwECAwb/QIBwSCwaj0hWQEBCOp8AFvOonEKvWNZghe0CSttk2EsulgzltLkgU5dvnIdbDZffTAHCqhYJsPl+bUM1HgF5XAA3hYYLiDQIJC0HAhlGNRKGDSgPd4eAaEoBG0QvfQIWB6MpGjMCIhUyShewM6JDMwcOMjQHdhxsNyhjMwMhGnBsRLwNMjcdTKutIRgxHKNRYwC1ujYQVtoBDUK1ybXX3XIA6OoQ6TQLgiUB6SXJQt32Woi1aGuCvP2QCYpiRUoTbIj03YARQU8iDgNUFBH4sN84W0LqCTI4kYNFjhWHUPxiT+NDQwIUjBjIixKRbtd4Xcs4Zt1De0NkChmp8147pp/9bHazaKSTxVYHkYr8Jcicz3RElLL7+M0pgJ49bV6tREOcEH1fx5jcyVRdBCtWiYBBNFQIr2wgpSJ1IacEAYkuEjRRSnFkvRgnDoJRUeMDiDEg1RaIYYKEwBoUOixuTDJezbOVzBpiIPEqps2CtCohcBAAIAYyzgwZG/UAabcHwsnA9XqO7XsTOrODerv3bdaJfQtXA+agi3nDk9PxEDvPCOUAggAAOw==)bi表示将一维矩阵组成二维矩阵函数。

不同Guass权重做差流程如下所示：
![](data:image/png;base64,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)

其中，所述频率自适应Mamba模块执行流程为：首先对特征进行归一化，紧接经过一个初始的线性嵌入层，然后分为两个分支：一个分支通过一个SiLU激活函数和深度可分离卷积层进入核心的SS2D模块，SS2D模块的输出经过层归一化；另一个分支通过一个FADC频率自适应空洞卷积、归一层以及一个SiLU激活；两个分支输出相加后再与最初的输入相加。

其中，FADC频率自适应空洞卷积卷积流程如下所示:
其中，![](data:image/png;base64,R0lGODlhJgAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAACgAmABMAhQAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAADo6ZjpmkDpmtjqQ22YAAGY6AGY6OmZmtmaQ22a222a2/5A6AJBmOpC2kJDb/7ZmALZmOraQZrbb/7b//9uQOtu2Ztu2kNvbttv///+2Zv/bkP/btv//tv//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwbjQIBwSCwKPw6IschBaJbQIobwjA5HjcWQxAgEDELRwSsZcgYba5GrFYYCBdDwEogMxWV1MSSwCEsTcX8TbUIXgnpEbGZoQhyIAFxgiUWHcgBvZSIIaXcHhQAlGAEEG1hwl2Z9QpKAeUOZRBkRfBUPIG+gmKuGBBS6mAGvsAEKrAyTsLwAHF9LsUy8knvLHMtEYrqAydnRfoaNRtOKDIXW33OI40uW2Ad568eFHc5QeETNWiMT4crfb16AMeoE4IIvLwlSyaNEZMoTQAKFYImox4ODCPeWNKnC8Nm1jgybFSCoJggAOw==)表示输出特征图中位置![](data:image/png;base64,R0lGODlhCwAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAAEAAKAA0AhAAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAADpmkDpmtjqQ22YAAGY6AGZmtma222a2/5A6AJBmOpC2kJDb/7ZmALZmOraQZrb//9uQOtv///+2Zv/bkP/btv//tv//2wECAwVFoFcFxMUxQZFZjiZITaYFCgDMid0tho0JEx0P4IH0bJtDbVfzAQHJh/CIoQE4kMHFRiFEAgFExlZs2s7Rs9oVVPtSW3cIADs=)处的像素值，![](data:image/png;base64,R0lGODlhDgAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADAAOAA0AhAAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAADo6kDpmtjqQtjqQ22YAAGa2/5A6AJA6OpDb/7ZmALZmOrbb/7b//9uQOtv///+2Zv/bkP/btv//tv//2wECAwECAwECAwECAwVQIABgRxAYwOaYAyUC12mJMTG9cNC8UTHjFQFEVEH8XiofQJI4vjQMw+aRwOFIJp31VdFlGMptpAUIDq1QJQllJS14ZGDgTZvLTTtVQKB4hQAAOw==)是卷积核大小，![](data:image/png;base64,R0lGODlhFQAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADAAUABEAhQAAAAAAAAAAOgAAZgA6OgA6kABmtjoAADoAOjoAZjo6OjqQkDqQtjqQ22YAAGYAOmYAZma222a2/5A6AJDb/7ZmALaQOrbb/7b/trb//9uQOtuQkNv/ttv///+2Zv/bkP//tv//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwaMwNAkECh0AIDPoYEEOYjGpmOQQQIqASZSObhYndHkIWv9IChWgDBcWTi0V0NaPYmCIJj39jxfHzUNTloVcFZ+dxmCSQlVc1dGGnKCQoVpFQUcD2iCHmGOlxZyAE4MExJpGmwEfKMOCp6OGgGirQJoSEK2Vhq6UrRbjI7CaR6/w5+Vx4amG43KywIRAEEAOw==)是卷积核的权重参数，![](data:image/png;base64,R0lGODlhWAAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADABXABEAhQAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAADoAOjoAZjo6Ojo6ZjpmZjpmkDpmtjqQ22YAAGYAOmY6AGY6OmZmtmaQ22a222a2/5A6AJBmOpC2kJC225Db25Db/7ZmALZmOraQZrbb/7b//9uQOtuQkNu2kNvbttv/ttv///+2Zv+2kP/bkP/btv//tv//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwb/QABAdQgELoCUcSASCk2TinNKrVqvUxemgMJ6R4uQdRR4CLWCzfRDEHvf8OnqIOjErSyJmTrfz5FOI0x3WCNccR4MB3uEUy0QjGcYBgBakX+NV4Zdby0RHFucmUIpdVSlHR6UUx6HZx8BBCJ5Aa6FtlgpXB6mlbCytLhUj5EAjwoOwxCrQiAWpRoUKErFVZtvlkllzc8C0dPbWK2ilRiDcotVSsnGy2/XXisIdo/M2uz1X71CIwpHp/+ojOiVj4oHIwgT1iLHj5mHcwAG2mlnT90+XSfcTVECyIkWZn3eCVPWkQygj05C5uolz87DJk44KmMkUSTDUxDnrCIWqBe8j5imHgGSmTIdOkAFv4xkZU/LIEwU39XR4mqOraT8trEwB1PpTSdCBf4jY2brIDQTWXHBEMCUC1jhhIxjRSCDkQZfrS3NqhAhJQ9174pakaCrEJ53oJarNipTNispKibZF0cQTMWNG2NmVQxxIzZiUGUeLbqKlgskYOZhHKfEBAtkChgeTSj27HICLPALMyUIADs=)表示输入特征图中位置![](data:image/png;base64,R0lGODlhCwAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAAEAAKAA0AhAAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAADpmkDpmtjqQ22YAAGY6AGZmtma222a2/5A6AJBmOpC2kJDb/7ZmALZmOraQZrb//9uQOtv///+2Zv/bkP/btv//tv//2wECAwVFoFcFxMUxQZFZjiZITaYFCgDMid0tho0JEx0P4IH0bJtDbVfzAQHJh/CIoQE4kMHFRiFEAgFExlZs2s7Rs9oVVPtSW3cIADs=)偏移![](data:image/png;base64,R0lGODlhGwAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADAAaABEAhQAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAADoAOjoAZjpmZjpmkDpmtjqQ22YAAGYAOmY6AGZmtma222a2/5A6AJBmOpC2kJDb25Db/7ZmALZmOraQZrb//9uQOtuQkNv///+2Zv/bkP/btv//tv//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwanQIAQQKIUPsOkcpkMHQQYppSZURwa06xw9LgYkVop6JiBCkmaAIEjggSOzCIWFMACNhKQwBL50O1KIQhRIw4GSXQMW4ZMHYdCGQMcQx1mAIWPSYUTlAGcRBSZToBDIJJDTo+FgJVRAB1wGZmgp06fmEubSh2er3UAIhSnRWa8AcfIx4cZBBXHC2AAIQmTWnJis1O2UhmkU3quSkUTHtVTvAXmQ8QSAEEAOw==)处的像素值，D为扩大感受野的空洞率。

S4:搭建Mamba序列建模模式协同空间-通道注意力的解码模块，以实现目标物体重建。

具体的，在本实施例中，所述目标物体重建模块执行流程为：第一阶段首先对特征进行层归一化，紧接经过一个初始的线性嵌入层，然后分为两个分支：一个分支通过一个SiLU激活函数和深度可分离卷积层进入核心的SS2D模块，SS2D模块的输出经过层归一化；另一个分支经过通道注意力模块，与前一个分支相乘后再与最初的输入相加；第二阶段先对第一阶段输出进行层归一化，紧接经过一个初始的线性嵌入层，然后分为两个分支：一个分支通过一个SiLU激活函数和深度可分离卷积层进入核心的SS2D模块，SS2D模块的输出经过层归一化；另一个分支经过空间注意力模块，与前一个分支相乘后再与第二阶段的原始输入相加；网络结构模型如图附图2c所示。

所述通道注意力模块包括并行的平均池化和最大池化、MLP模块以及Sigmoid激活函数。

所述空间注意力模块包括并行的平均池化和最大池化、卷积核为7*7的卷积层以及Sigmoid激活函数。

S5:采用UNet架构，联合子任务模块，构建出整个网络模型，完成双时图像到二值变化图的转换。

具体的，在本实施例中，所述整个网络模型的执行流程为：首先经过线性嵌入模块后得到尺寸为H×W×C1的特征信息；进入第一层堆叠的两个编码模块，通过下采样获得尺寸为H/2×W/2×C2的特征信息；传入第二层堆叠的两个编码模块，再经过下采样获得尺寸为H/4×W/4×C3的特征信息；传入第三层堆叠的十个编码模块；第三层编码信息经过第三层解码模块和差异捕获模块后，通过动态融合模块融合并输出解码信息，通过上采样层获得尺寸为H/2×W/2×C2的特征信息；第二层和第一层解码模块和差异捕获模块在第三层的基础上，输出尺寸为 H×W×C1的编码信息；最后通过分类器模块将编码信息映射成二值变化图。网络结构模型如图附图2d所示。

S6: 采用交叉熵与Lovasz-softmax 的联合损失训练所属网络模型，以使得训练结果更加稳定。

具体的，在本实施例中，所述联合损失具体为：

S61：建立所述交叉熵损失函数：

![](data:image/png;base64,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)

其中![](data:image/png;base64,R0lGODlhEAAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAAEAAPAA0AhAAAAAAAAAAAOgAAZgA6kABmZgBmtjoAADoAZjpmtjqQ22YAAGYAZmZmtmaQ22a222a2/5A6AJA6ZpDb/7ZmALZmOrb//9uQOtuQkNu2Ztv///+2Zv/bkP//tv//2wECAwVXIJAdwgQAWxBA58kl3WK01GC1OEVoJ6XguIsN0GHcgKdNCfVDug4Qj+QIuOxaMcWG5Tx5IoVG67vMDVscBPVU4bY2M9zGgfSxJ5kE8gvBWCICdHuADwAhADs=)表示第![](data:image/png;base64,R0lGODlhBwAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAEADAAFAA0AhAAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAAGa2/5A6AJDb/7ZmALZmOrb//9uQOtv/29v///+2Zv/bkP//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwUmIAA4giIuQyOu7BgEiOikZwEBU2KI0hEDkdKJBlgUHgxF5EAwhQAAOw==)个像素中的真值,![](data:image/png;base64,R0lGODlhDwAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADAAPABEAhQAAAAAAAAAAOgAAZgA6ZgA6kABmkABmtjoAADoAOjoAZjo6kDpmtjqQ22YAAGYAOma2tma222a2/5A6AJA6OpBmOpDb/7ZmALZmOrb//9uQOtuQkNv///+2Zv/bkP//tv//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwZxQIDQgyhwhMgk4OOATA5KJUYCAFEs0awWMAkEBhlQNwBNisFCcVmpEWABnsRbyWwINevoBfx5zKMdARIdeQAaRmkTBhNUW4YBiIluSUSNSB4KGZQLR0mEQlcAF3ZKo0ITDBWRSGISGxldBJpKYgIRAEEAOw==)表示第![](data:image/png;base64,R0lGODlhBwAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAEADAAFAA0AhAAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAAGa2/5A6AJDb/7ZmALZmOrb//9uQOtv/29v///+2Zv/bkP//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwUmIAA4giIuQyOu7BgEiOikZwEBU2KI0hEDkdKJBlgUHgxF5EAwhQAAOw==)个像素的概率；![](data:image/png;base64,R0lGODlhDgAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADAAOAA0AhAAAAAAAAAAAOgAAZgA6kABmZgBmtjoAADoAOjoAZjo6kDqQ22YAAGYAOma2/5A6AJBmAJDb/7ZmALb//9uQOtv///+2Zv/bkP//tv//2wECAwECAwECAwECAwECAwECAwVXIABcRzBMYvYEgSMCFuuKF7O8cMEY74VEOMpCchJZCJWXyhGbUXgvTCOC2QFUt54iSZxIga+nsXVJoF6SbNVwTIq+6AEkSzO/Yq1g8c0QgAEULEhogyIhADs=)表示像素数；![](data:image/png;base64,R0lGODlhGwAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADAAZABEAhQAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAADoAOjoAZjo6ZjpmtjqQtjqQ22YAAGY6AGY6OmaQ22a222a2/5A6AJA6OpA6ZpBmOpC225C2/5Db/7ZmALZmOrbb/7b//9uQOtu2Ztu2kNvbttv///+2Zv/bkP/btv//tv//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwaMQIBQiOIEBJGhcskEoCiG0UegaVqHpMAEcIJkrtcNFUx+FkZk8MlhSINLh4b7+tHOrZuB585cy4coFh13G2eAG21uJhUBjY4BekIhD41bACWUVBUSQiIXBwEEgwAkAhgoFVWlGAAbf1dPlkQUciAIVW8Je0NwjQqsZCW3AB9yJQlVIAtoZEZIQ84MaEEAOw==)为交叉熵损失函数。

S62：建立交叉熵与Lovasz-softmax联合损失函数：

![](data:image/png;base64,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)

其中![](data:image/png;base64,R0lGODlhCgAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADAAKAA0AhAAAAAAAAAAAOgAAZgA6kABmtjoAADoAOjpmtjqQ22YAAGY6AGaQ22a2/5A6AJA6OpBmZpDb/7ZmALZmOrbb/7b//9uQOtv/29v///+2Zv/bkP/btv//tv//2wECAwECAwU9ICBuS0BgothBlHZEadw5TRxLtQ3MuS0FvVQmALRxFAWcj3B5wFIWQaTjTGkMiR1NdBxUtA0N49ebBBCYEAA7)为损失函数联合权重；![](data:image/png;base64,R0lGODlhIQAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADAAgABEAhQAAAAAAAAAAOgAAZgA6ZgA6kABmtjoAADoAOjoAZjpmkDqQ22YAAGYAOmYAZmY6AGY6OmY6ZmaQ22a222a2/5A6AJA6OpA6ZpBmOpC2/5Db/7ZmALZmOrbbtrbb/7b//9uQOtuQZtuQkNu2Ztu2ttvbttv/ttv///+2Zv/bkP/btv//tv//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwaqQIBQyOIEBJKhcslsAlgVwwkk0DivTlSAAlhBMtiwclMVm4fQwul8XjEM7HPqsIibQVu7eDP4MLVwem51aBceABuEdhtqaBtwUFxxKhYBlpcBfV0OfkIjBwEEViwYVSoPBR8WE0IlGKAEhwAojbNlHGokHQ0aJCacYiCBbpIgjSkJfiiKV5FCKQhWiIG0JywXnVgrvCIf0FxU0sIqEdJhUAKSIbCyswEK7UEAOw==)为Lovasz-softmax损失函数；![](data:image/png;base64,R0lGODlhLQAfAHcAMSH+GlNvZnR3YXJlOiBNaWNyb3NvZnQgT2ZmaWNlACH5BAEAAAAALAAADAAsABEAhQAAAAAAAAAAOgAAZgA6OgA6ZgA6kABmtjoAADoAOjoAZjo6ZjpmkDqQ22YAAGYAOmYAZmY6OmZmtmaQ22a222a2/5A6AJA6OpA6ZpBmOpCQtpC225C2/5Db/7ZmALZmOrbb/7b/27b//9uQOtuQZtuQkNu2Ztvbttv///+2Zv/bkP/btv//tv//2wECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwECAwbZQIBQ2PoEBJOhcslsOpctywE1EnSe2GwzFagAWBGOdqz1WMnobNSASrubLMfhTVeqEI26ftTV1z0DIn5aXHNCcXlDLRggACmGWI9liQAebIoec1GUTptqFl4rFwGkpQGBcaWJJggBBVeppKutBAuCLBCCFxRCJxmtBY0AKgqCQilnH5fExo4CGwAjhimXhNVxXtGX1G0AUdkeidJk4UMqCVeVhuVC57APV99j3ysSKOdeVfGg9ffoKxkUhNDAAl6JZk+MFDBGApgwAAoFtcgQgMEJB8GiCPASBAA7) 为联合损失函数。

具体的，本实施例中对Botswana数据集利用本发明的一种基于Mamba模型的遥感图像变化检测方法进行了实验，实验的结果如表1所示：


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

版本命名
Siamese   indepence 
编码器：
1. 编码器跑一个完整的 en_sicnn_inmlp   de_ca_sa  di_so_fo_m  
2. 全孪生 en_sicnn_simlp
3. 全非孪生 en_incnn_inmlp
解码器 
4. 只留   de
5. 留空间  de_sa
6. 留通道 de_ca
差异
7. 去1阶 di_so_m
8. 去2阶 di_fo_m
9. 全去了 di_m
10. 把mamba拿掉 di_so_fo



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
62. 106714 1577 v17  提交了两个一样的 第一轮效果还行9075 第二轮超时
63. 106695 1485 v15 最好的一版 把验证集换成测试集 反而掉点了 9045 
64. 106716 1486 v17 val改test 弄错了
65. 106817 1485 v17 把edge里面的两个fuse改成原来的了
66. 106873 1486 v17 val改test 老版edgefuse
67. 106896 1577 v17 
68. 106897 1577 v17
69. 106898 1577 v17
70. 106899 1577 v17
71. 106900 1577 v17
72. 106901 1577 v17
73. 106920 1486 V17 test 0.7 300 少了个文件
74. 106970 1486 V17 test 0.7 300
75. 106971 1486 V17 test 0.7 300
76. 106972 1486 V17 test 0.7 300

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

# net
需求
1. 脚本传入参数
	1. 控制模型结构，来消融，
	2. 控制超参，只要想再改动的内容都要放到参数里
	3. 控制数据集
2. 输出
	1. 权重
	2. loss图 只用版本号
	3. out 直接在.sh中控制

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