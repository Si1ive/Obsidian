# 环境配置
## 常规版本选择
 1. cuda 11.8
 2. pytorch 2.1.1
 3. python 3.10
 ## 常规版本安装
 1. conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

## Linux
使用wsl2，ubuntu20.04


## Conda指令
1. conda create -n name python=
2. conda activate
3. conda deactivate
4. conda create -n newenv --clone oldenv
5. conda env list

## wsl指令
 1. wsl -l -v

## 路径问题
 1. 如果是跑linux里的代码，那么其中的绝对路径就按linux的地址解析（例如‘/usr/local/…’），写‘ //wsl$/Ubuntu/usr/local/…’反而找不到
 2. 如果是跑windows下的代码，其中的绝对路径就按windows的地址格式解析（例如‘ //wsl$/Ubuntu/usr/local/…’），如果还写‘/usr/local/…’就会找不到
## Causal-conv1d报错
版本1.4.0

## Mamba-ssm报错
ImportError: /home/zzh/anaconda3/envs/cdmamba/lib/python3.10/site-packages/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops10zeros_like4callERKNS_6TensorESt8optionalIN3c1010ScalarTypeEES5_INS6_6LayoutEES5_INS6_6DeviceEES5_IbES5_INS6_12MemoryFormatEE
版本应该换成1.2.0.post1

## Torch指令
1. import torch
2. torch.__version__
3. torch.cuda.is_available()
4. 