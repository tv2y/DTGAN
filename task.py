import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pre_processing
from Network_Model import Generator, Discriminator, Classifier
from torch.utils import data as Data
import sys
import wandb
import logging
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from experiments.WGAN.problem import Toy
from experiments.WGAN.utils import plot_2d_pareto
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    set_logger,
)
from methods.weight_methods import WeightMethods
sys.path.append("../..")
set_logger()

start = time.time()
pre_processing.setup_seed(pre_processing.seed)
writer = SummaryWriter("logs")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建网络
generator = Generator().to(device)
discriminator = Discriminator().to(device)

def task_d(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader, adv_normal_trainset_dataloader, adv_attack_trainset_dataloader):
    d_data_length = 0
    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    loss4 = 0.0
    for _, data in enumerate(zip(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader, adv_normal_trainset_dataloader, adv_attack_trainset_dataloader)):

        benign_normal_data = data[0][0].to(device)
        benign_normal_target = data[0][1].to(device)
        benign_attack_data = data[1][0].to(device)
        benign_attack_target = data[1][1].to(device)
        adv_normal_data = data[2][0].to(device)
        adv_normal_target = data[2][1].to(device)
        adv_attack_data = data[3][0].to(device)
        acv_attack_target = data[3][1].to(device)

        # 训练模型D
        benign_normal_after_generator = generator(benign_normal_data).detach().to(device)
        benign_attack_after_generator = generator(benign_attack_data).detach().to(device)
        adv_normal_after_generator = generator(adv_normal_data).detach().to(device)
        adv_attack_after_generator = generator(adv_attack_data).detach().to(device)

        d_data_length += len(data)

        d_benign_normal_output = discriminator(benign_normal_data).to(device)
        d_benign_attack_output = discriminator(benign_attack_data).to(device)
        d_benign_normal_after_generator_output = discriminator(benign_normal_after_generator).to(device)
        d_benign_attack_after_generator_output = discriminator(benign_attack_after_generator).to(device)
        d_adv_normal_after_generator_output = discriminator(adv_normal_after_generator).to(device)
        d_adv_attack_after_generator_output = discriminator(adv_attack_after_generator).to(device)

        d_g_benign_nomal_loss = torch.mean(d_benign_normal_after_generator_output) - torch.mean(d_benign_normal_output)
        d_g_benign_attack_loss = torch.mean(d_benign_attack_after_generator_output) - torch.mean(d_benign_attack_output)
        d_g_adv_normal_loss = torch.mean(d_adv_normal_after_generator_output) - torch.mean(d_benign_normal_output)
        d_g_adv_attack_loss = torch.mean(d_adv_attack_after_generator_output) - torch.mean(d_benign_attack_output)
        # d_g_adv_normal_loss = -torch.mean(d_adv_normal_after_generator_output) + torch.mean(d_benign_normal_output)
        # d_g_adv_attack_loss = -torch.mean(d_adv_attack_after_generator_output) + torch.mean(d_benign_attack_output)

        loss1 += d_g_benign_nomal_loss
        loss2 += d_g_benign_attack_loss
        loss3 += d_g_adv_normal_loss
        loss4 += d_g_adv_attack_loss

    loss1 = loss1 / d_data_length
    loss2 = loss2 / d_data_length
    loss3 = loss3 / d_data_length
    loss4 = loss4 / d_data_length

    loss_d = torch.stack([loss3, loss4]) * 1

    return loss_d


def task_g(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader, adv_normal_trainset_dataloader, adv_attack_trainset_dataloader):
    g_data_length = 0
    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    loss4 = 0.0
    for _, data in enumerate(zip(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader,  adv_normal_trainset_dataloader, adv_attack_trainset_dataloader)):
        benign_normal_data = data[0][0].to(device)
        benign_normal_target = data[0][1].to(device)
        benign_attack_data = data[1][0].to(device)
        benign_attack_target = data[1][1].to(device)
        adv_normal_data = data[2][0].to(device)
        adv_normal_target = data[2][1].to(device)
        adv_attack_data = data[3][0].to(device)
        acv_attack_target = data[3][1].to(device)

        g_data_length += len(data)

        benign_normal_after_generator = generator(benign_normal_data).to(device)
        benign_attack_after_generator = generator(benign_attack_data).to(device)
        adv_normal_after_generator = generator(adv_normal_data).to(device)
        adv_attack_after_generator = generator(adv_attack_data).to(device)

        d_benign_normal_output = discriminator(benign_normal_after_generator).to(device)
        d_benign_attack_output = discriminator(benign_attack_after_generator).to(device)
        d_adv_normal_output = discriminator(adv_normal_after_generator).to(device)
        d_adv_attack_output = discriminator(adv_attack_after_generator).to(device)

        # c_benign_normal_output = classifier(benign_normal_after_generator).softmax(dim=-1)
        # c_benign_attack_output = classifier(benign_attack_after_generator).softmax(dim=-1)
        # c_g_benign_normal_loss = c_g_loss_fn(c_benign_normal_output, benign_normal_target)
        # c_g_benign_attack_loss = c_g_loss_fn(c_benign_attack_output, benign_attack_target)

        d_g_benign_nomal_loss = -torch.mean(d_benign_normal_output)
        d_g_benign_attack_loss = -torch.mean(d_benign_attack_output)
        d_g_adv_normal_loss = -torch.mean(d_adv_normal_output)
        d_g_adv_attack_loss = -torch.mean(d_adv_attack_output)
        # d_g_adv_normal_loss = torch.mean(d_adv_normal_output)
        # d_g_adv_attack_loss = torch.mean(d_adv_attack_output)

        loss1 += d_g_benign_nomal_loss
        loss2 += d_g_benign_attack_loss
        loss3 += d_g_adv_normal_loss
        loss4 += d_g_adv_attack_loss

    loss1 = loss1 / g_data_length
    loss2 = loss2 / g_data_length
    loss3 = loss3 / g_data_length
    loss4 = loss4 / g_data_length

    loss_g = torch.stack([loss3, loss4]) * 1

    return loss_g

def train(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader, adv_normal_trainset_dataloader, adv_attack_trainset_dataloader):

    #初始化模式
    parser = ArgumentParser(
        "Toy example (modification of the one in CAGrad)", parents=[common_parser]
    )
    # parser.set_defaults(n_epochs=35000, method="nashmtl", data_path=None)
    parser.set_defaults(n_epochs=300, method="nashmtl", data_path=None)
    parser.add_argument(
        "--scale", default=1e-1, type=float, help="scale for first loss"
    )
    parser.add_argument("--out-path", default="outputs", type=Path, help="output path")
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    args = parser.parse_args()

    weight_methods_parameters = extract_weight_method_parameters_from_args(args)    # 获取模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_iter = args.n_epochs
    n_tasks = 2     #任务数量

    #定义优化器
    method_d = WeightMethods(
        method=args.method,
        device=device,
        n_tasks=n_tasks,
        **weight_methods_parameters[args.method],
    )
    optimizer_d = torch.optim.Adam(
        [
            dict(params=discriminator.parameters(), lr=1e-4),
            dict(params=method_d.parameters(), lr=args.method_params_lr),
        ]
    )

    method_g = WeightMethods(
        method=args.method,
        device=device,
        n_tasks=n_tasks,
        **weight_methods_parameters[args.method],
    )
    optimizer_g = torch.optim.RMSprop(
        [
            dict(params=generator.parameters(), lr=1e-4),
            dict(params=method_g.parameters(), lr=args.method_params_lr),
        ]
    )


    for _ in range(n_iter):

        optimizer_d.zero_grad()
        loss_d = task_d(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader, adv_normal_trainset_dataloader, adv_attack_trainset_dataloader)
        _d = method_d.backward(
            losses=loss_d,
            shared_parameters=list(discriminator.parameters()),
            task_specific_parameters=None,
            last_shared_parameters=None,
            representation=None
        )
        optimizer_d.step()

        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        optimizer_g.zero_grad()
        loss_g = task_g(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader, adv_normal_trainset_dataloader, adv_attack_trainset_dataloader)

        _g = method_g.backward(
            losses=loss_g,
            shared_parameters=list(generator.parameters()),
            task_specific_parameters=None,
            last_shared_parameters=None,
            representation=None
        )
        optimizer_g.step()

        print("epoch:", _, "\n",_d,"\n",loss_d,"\n",loss_g,"\n")

    return generator
