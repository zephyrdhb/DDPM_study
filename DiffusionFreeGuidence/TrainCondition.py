import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler

# 这两个列表为定义的全局列表，用于保存梯度信息
grad_inputs = []
grad_ouputs = []


def backward_hook(module, grad_input, grad_ouput):
    grad_inputs.append(grad_input)
    grad_ouputs.append(grad_ouput)


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


# 定义一个获取权重梯度l2范数的函数，包括获取层名及该层权重的梯度
def get_params_grad_l2(model):
    params_names = []
    params_grads = []
    for name, parms in model.named_parameters():
        params_names.append(name)
        params_grads.append(parms.grad.norm())
    return params_names, params_grads


def train(modelConfig: Dict):
    # multi GPUs setting
    local_rank = modelConfig['local_rank']
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', world_size=1, rank=local_rank)
    torch.cuda.set_device(local_rank)

    # dataset
    dataset = CIFAR10(
        root='D:/数据集/Cifar10', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], num_workers=4, drop_last=True, pin_memory=True,
        sampler=train_sampler)

    # model setup
    net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"])), strict=False)
        print("Model weight load down.")
    net_model = net_model.cuda(local_rank)
    net_model = torch.nn.parallel.DistributedDataParallel(net_model, device_ids=[local_rank], )
    # 这块为注册内容，需放在模型构建完成后、开始训练前。
    for i, v in net_model.module.named_children():
        for j, v1 in v.named_children():
            v1.register_backward_hook(hook=backward_hook)

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).cuda(local_rank)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).cuda(non_blocking=True)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])

                # 获取权重的梯度
                params_names, params_grads = get_params_grad_l2(net_model)

                # 需要删除中间变量梯度 释放内存
                clear_hook_grad()

                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.module.state_dict(), os.path.join(
            modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # multi GPUs setting
    local_rank = modelConfig['local_rank']
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', world_size=1, rank=local_rank)
    torch.cuda.set_device(local_rank)

    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)
        labelList = []
        k = 0
        for i in range(1, modelConfig["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).cuda(non_blocking=True) + 1
        print("labels: ", labels)
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]))
        model.load_state_dict(ckpt)
        print("model load weight done.")

        model = model.cuda(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], )

        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).cuda(local_rank)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]],
            device=f'cuda:{local_rank}')
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        print(sampledImgs)
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
