# DenoisingDiffusionProbabilityModel
由项目 https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm- 修改而来 <br>
<br>
添加了并行运算、获取中间变量(即隐层特征表示)梯度信息、获取中间层权重梯度信息。
<br>
<br>
## 使用torch.distributed模块将代码改造成并行计算

#####  1.直接使用torch.distributed 必须通过执行如下命令才能启动并行运算。

``` bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env MainCondition.py
```

##### 2.使用 torch.multiprocessing 取代启动器，在代码中添加

```python
import torch.multiprocessing as mp
import argparse

""" 
其中main为需要运行的主函数 格式为 main(local_rank, nprocs, args)
local_rank指当前的第几个GPU(在同一台机子上), nprocs为GPU数量,distributed会自动在args中传入如local_rank之类的参数，使用torch.multiprocessing 则可以直接在main函数中传入，可以省略args，并且也不需要使用如1中所示的bash命令。
"""
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()
mp.spawn(main, nprocs=1, args=(1, args))
```

##### 3.在mian函数中需要加入如下代码，用来定义当前分布式环境的一些参数。

```python
import torch.distributed as dist

local_rank = modelConfig['local_rank']
dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', world_size=1,rank=local_rank)
torch.cuda.set_device(local_rank)
```

##### 4.除了以上步骤外，还需要对模型、数据集进行分布式处理。需要注意的是，模型保存权重时，需使用model.module.xxx来保存。

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(
dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True,sampler=train_sampler)

net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])
net_model = net_model.cuda(local_rank)
net_model = torch.nn.parallel.DistributedDataParallel(net_model, device_ids=[local_rank],
                                                          output_device=[local_rank])
```

## 实现函数,整合进代码框架中, 功能为可以选择并查看反向传播过程中的非叶子张量(计算图的中间变量)的梯度(提升:可以使用torch.nn中的Hook方法).



##### 1.在模型构建完成后，但是在模型需要输入数据前，对模型中特定层使用register_backward_hook注册反向传播的hook，并简历hook函数。需要将该注册步骤放置于TrainCondition.py文件中。

```python
model = UNet(xxx)

# 由于model中有双层封装，因此通过两个循环遍历读取里面的block，给各个block层注册反向传播hook。
# 这块为注册内容，需放在模型构建完成后、开始训练前。
for i, v in model.named_children():
	for j, v1 in v.named_children():
		v1.register_backward_hook(hook=backward_hook)
        
# 这两个列表为定义的全局列表，用于保存梯度信息
grad_inputs = []
grad_ouputs = []

# 反向传播Hook函数，参数格式必须为(module, grad_input, grad_ouput)
# 其中module为该层module，grad_input为该层输入变量的梯度，grad_ouput为该层输出变量的梯度。
# 注意grad_input和grad_ouput是tuple形式，因为很多情况下每层的输入变量个数不为1。

def backward_hook(module, grad_input, grad_ouput):
    print("hooker working")
    grad_inputs.append(grad_input)
    grad_ouputs.append(grad_ouput)
```

##### 其中需要注意的是在每次backward完要开始下一次迭代训练前，需要运行clear_hook_grad()将本次的梯度信息释放，否则显存会无限占用。

```python
# 清除本次迭代中间变量的梯度信息 释放内存
def clear_hook_grad():
    for i in grad_inputs:
        for ii in i:
            if ii is not None:
                ii.detach().cpu()
    for i in grad_ouputs:
        for ii in i:
            if ii is not None:
                ii.detach().cpu()
    del grad_inputs[:]
    del grad_ouputs[:]
```

<div style="page-break-after: always;"></div>

##### 2.主要代码如图

![image-20230217220919326](C:\Users\z\AppData\Roaming\Typora\typora-user-images\image-20230217220919326.png)

##### 3.运行结果 其中包含了各个层的输入变量的梯度信息，输出变量的梯度信息同理

![image-20230217223058203](C:\Users\z\AppData\Roaming\Typora\typora-user-images\image-20230217223058203.png)

## 实现函数,整合进代码框架中, 功能为可以选择并查看任意一步反向传播中**模型参数的梯度**(例如某个线性层的梯度,卷积层的梯度),并返回梯度的L2 norm.

##### 1.在loss.backward()进行反向传播计算梯度后，可以直接通过遍历models.named_parameters()，使用 params.grad获得各层参数的梯度。

```python
# 定义一个获取权重梯度l2范数的函数，包括获取层名及该层权重的梯度
def get_params_grad_l2(model):
    params_names = []
    params_grads = []
    for name, parms in model.named_parameters():
        params_names.append(name)
        params_grads.append(parms.grad.norm())
    return params_names,params_grads
```

##### 2.运行结果 params_grads包含了各层权重的梯度信息的l2范数

![image-20230217223154115](C:\Users\z\AppData\Roaming\Typora\typora-user-images\image-20230217223154115.png)