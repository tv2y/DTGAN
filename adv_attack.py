import torch
import torchattacks
from torch.utils import data as Data
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def fgsm(dataloader, model, eps):
    x = dataloader.dataset.data.to(device)
    y = dataloader.dataset.target.to(device)
    label = torch.argmax(y, dim=1).to(device)
    attack = torchattacks.FGSM(model, eps=eps)
    adv_x = attack(x, label)
    # 打包成dataloader
    dataset = Data.TensorDataset(adv_x, y)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    # 加载数据
    adv_dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False)
    return adv_dataloader


def bim(dataloader, model, eps=8 / 255, alpha=2 / 255, steps=10):
    x = dataloader.dataset.data.to(device)
    y = dataloader.dataset.target.to(device)
    label = torch.argmax(y, dim=1).to(device)
    attack = torchattacks.BIM(model, eps=eps, alpha=alpha, steps=steps)
    adv_x = attack(x, label)
    # 打包成dataloader
    dataset = Data.TensorDataset(adv_x, y)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    # 加载数据
    adv_dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False)
    return adv_dataloader


def pgd(dataloader, model, eps=8 / 255, alpha=2 / 255, steps=10):
    x = dataloader.dataset.data.to(device)
    y = dataloader.dataset.target.to(device)
    label = torch.argmax(y, dim=1).to(device)
    attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    adv_x = attack(x, label)
    # 打包成dataloader
    dataset = Data.TensorDataset(adv_x, y)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    # 加载数据
    adv_dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False)
    return adv_dataloader


def jsma(dataloader, model, theta=1.0, gamma=0.1):
    x = dataloader.dataset.data.to(device)
    y = dataloader.dataset.target.to(device)
    label = torch.argmax(y, dim=1).to(device)
    attack = torchattacks.JSMA(model, theta=theta, gamma=gamma)
    attack.set_mode_targeted_by_label(quiet=True)
    target_labels = (label + 1) % 2
    adv_x = attack(x, target_labels)
    # 打包成dataloader
    dataset = Data.TensorDataset(adv_x, y)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    # 加载数据
    adv_dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False)
    return adv_dataloader


def cw(dataloader, model, c=1, kappa=0, steps=50, lr=0.01):
    x = dataloader.dataset.data.to(device)
    y = dataloader.dataset.target.to(device)
    label = torch.argmax(y, dim=1).to(device)
    attack = torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)
    attack.set_mode_targeted_by_label(quiet=True)
    target_labels = (label + 1) % 2
    adv_x = attack(x, target_labels)
    # 打包成dataloader
    dataset = Data.TensorDataset(adv_x, y)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    # 加载数据
    adv_dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False)
    return adv_dataloader

def deepfool(dataloader, model, steps=50, overshoot=0.02):
    x = dataloader.dataset.data.to(device)
    y = dataloader.dataset.target.to(device)
    label = torch.argmax(y, dim=1).to(device)
    attack = torchattacks.DeepFool(model, steps=steps,overshoot=overshoot)
    # attack.set_mode_targeted_by_label(quiet=False)
    # target_labels = (label + 1) % 2
    target_labels = label
    adv_x = attack(x, target_labels)
    # 打包成dataloader
    dataset = Data.TensorDataset(adv_x, y)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    # 加载数据
    adv_dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False)
    return adv_dataloader

def gn(dataloader, model, std=0.1):
    x = dataloader.dataset.data.to(device)
    y = dataloader.dataset.target.to(device)
    label = torch.argmax(y, dim=1).to(device)
    attack = torchattacks.GN(model, std=std)
    adv_x = attack(x, label)
    # 打包成dataloader
    dataset = Data.TensorDataset(adv_x, y)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    # 加载数据
    adv_dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False)
    return adv_dataloader

def jitter(dataloader, model, eps=0.2, alpha=0.02, steps=50, scale=10, std=0.1, random_start=True):
    x = dataloader.dataset.data.to(device)
    y = dataloader.dataset.target.to(device)
    label = torch.argmax(y, dim=1).to(device)
    attack = torchattacks.Jitter(model, eps=eps, alpha=alpha, steps=steps, scale=scale, std=std, random_start=random_start)
    adv_x = attack(x, label)
    # 打包成dataloader
    dataset = Data.TensorDataset(adv_x, y)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    # 加载数据
    adv_dataloader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False)
    return adv_dataloader

def attack(attack_type, attack_parameter_dict, dataloader, model):
    dict = attack_parameter_dict
    if attack_type == "fgsm":
                return fgsm(dataloader, model, dict['eps'])
    elif attack_type == "bim":
        return bim(dataloader, model, dict['eps'], dict['alpha'], dict['steps'])
    elif attack_type == "pgd":
        return pgd(dataloader, model,dict['eps'], dict['alpha'], dict['steps'])
    elif attack_type == "jsma":
        return jsma(dataloader, model,dict['theta'], dict['gamma'])
    elif attack_type == "cw":
        return cw(dataloader, model, dict['c'], dict['kappa'], dict['steps'], dict['lr'])
    elif attack_type == "deepfool":
        return deepfool(dataloader, model, dict['steps'], dict['overshoot'])
    elif attack_type == "gn":
        return gn(dataloader, model, dict['std'])
    elif attack_type == "jitter":
        return jitter(dataloader, model, dict['eps'], dict['alpha'], dict['steps'], dict['scale'], dict['std'], dict['random_start'])
    else:
        return

