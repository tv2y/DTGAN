import sys
import time

import numpy

import pre_processing
import torch
import train_Adv_WGAN
from torch.utils import data as Data
from torch.utils.data import DataLoader
import adv_attack
import task
from pandas.errors import SettingWithCopyWarning

# 1、训练二分类器
def train_binary_classification():
    epoch = 100
    train_dataloader = pre_processing.Load_Trainset_To_1_Dataloader()
    classifier = pre_processing.train_classifier_on_trainset(train_dataloader, epoch)
    torch.save(classifier, r"model\classifier\NSL_KDD_binary_classifier_"+str(time.time())+".pth")
    test_dataloader = pre_processing.Load_Testset_To_1_Dataloader()
    pre_processing.test_classifier_on_test(train_dataloader, classifier)
    pre_processing.test_classifier_on_test(test_dataloader, classifier)
    return classifier


if __name__ == "__main__":
    attack_type = ["fgsm", "bim", "pgd", "jsma", "cw","deepfool","gn","jitter"]
    attack_parameter_dict = {"fgsm": {"no": 0, "eps": 0.2},
                             "bim": {"no": 1, "eps": 0.2, "alpha": 0.02, "steps": 50},
                             "pgd": {"no": 2, "eps": 0.2, "alpha": 0.02, "steps": 50},
                             "jsma": {"no": 3, "theta": 1.0, "gamma": 0.1},
                             "cw": {"no": 4, "c": 1, "kappa": 0, "steps": 50, "lr": 0.05},
                             "deepfool":{"no": 5, "steps": 10, "overshoot": 2}}
    # 用于测试的攻击类型
    attack_type_num = 0

    start_time = time.time()
    pre_processing.setup_seed(pre_processing.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    """
    一、训练二分类器
    """
    print("一、训练二分类器")
    # classifier = train_binary_classification()
    classifier = torch.load(r"D:\tv2y\code\experiment2\Experiment_for_NSL_KDD\model\classifier\NSL_KDD_binary_classifier_1710399440.29307.pth") # GPU

    """
    二、生成对抗样本
    """
    print("二、生成对抗样本")
    # 将训练集中的数据加载为attack、normal两类
    benign_attack_trainset_dataloader = pre_processing.Load_One_Type_Trainset("attack")
    benign_normal_trainset_dataloader = pre_processing.Load_One_Type_Trainset("normal")

    # 将训练集中的攻击类型流量利用攻击算法生成相应的对抗样本
    adv_attack_trainset_dataloader = adv_attack.attack(attack_type[attack_type_num], attack_parameter_dict[attack_type[attack_type_num]], benign_attack_trainset_dataloader, classifier)
    adv_normal_trainset_dataloader = adv_attack.attack(attack_type[attack_type_num], attack_parameter_dict[attack_type[attack_type_num]], benign_normal_trainset_dataloader, classifier)

    """
    三、训练Adv_WGAN模型
    """
    print("三、训练Adv_WGAN模型")
    generator = train_Adv_WGAN.Train_Adv_WGAN(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader, adv_normal_trainset_dataloader, adv_attack_trainset_dataloader, a=0.4, b=0.2, c=0.1, d=0.3)
    torch.save(generator, r"model\generator\generator_" + str(start_time) + ".pth")

    # 使用预先准备的生成器
    # generator = torch.load(r"D:\tv2y\code\experiment2\Experiment_for_NSL_KDD\model\generator\generator_1711442232.428504.pth")

    """
    四、检验Adv_WGAN模型防御对抗样本攻击效果
    """
    print("四、检验Adv_WGAN模型防御对抗样本攻击效果")

    # ②检验分类器对各阶段训练集数据的分类性能
    print("————————————————针对训练集数据————————————————")
    # 初始阶段：二分类器对原始训练集的二分类性能
    print("初始阶段：二分类器对原始训练集的二分类性能")
    train_dataloader = pre_processing.Load_Trainset_To_1_Dataloader()
    pre_processing.test_classifier_on_test(train_dataloader, classifier)
    # 生成对抗样本阶段：二分类器在对抗样本攻击情况下的二分类性能
    print("生成对抗样本阶段：二分类器在对抗样本攻击情况下的二分类性能")
    adv_all_trainset_dataloader = pre_processing.Merge_Two_Dataloader_To_One(adv_attack_trainset_dataloader,adv_normal_trainset_dataloader)
    attack_dataloader = pre_processing.Merge_Two_Dataloader_To_One(adv_all_trainset_dataloader, train_dataloader)
    # attack_dataloader = adv_all_trainset_dataloader
    pre_processing.test_classifier_on_test(attack_dataloader, classifier)
    # 防御阶段：二分类器对经生成器还原的干净样本的二分类性能
    print("防御阶段：二分类器对经生成器还原的干净样本的二分类性能")
    clean_data = generator(attack_dataloader.dataset.data)
    dataset = Data.TensorDataset(clean_data, attack_dataloader.dataset.target)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    clean_dataloader = DataLoader(dataset=dataset, batch_size=pre_processing.batch_size, shuffle=False)
    pre_processing.test_classifier_on_test(clean_dataloader, classifier)


    # ③在测试集上做仿真实验，检验真实情况中针对恶意流量检测的对抗样本攻击防御效果
    print("————————————————针对测试集数据————————————————")
    # 首先检验二分类器对测试集数据的初始分类性能
    print("初始阶段：二分类器对原始测试集的二分类性能")
    test_dataloader = pre_processing.Load_Testset_To_1_Dataloader()
    pre_processing.test_classifier_on_test(test_dataloader, classifier)
    # 针对测试集中的攻击类型数据生成对抗样本，检验二分类器在对抗样本攻击情况下的性能
    print("生成对抗样本阶段：二分类器在对抗样本攻击情况下的二分类性能")
    benign_attack_testset_dataloader = pre_processing.Load_One_Type_Testset("attack")
    benign_normal_testset_dataloader = pre_processing.Load_One_Type_Testset("normal")

    adv_all_testset_dataloader = adv_attack.attack(attack_type[attack_type_num], attack_parameter_dict[attack_type[attack_type_num]], test_dataloader, classifier)
    # adv_all_testset_dataloader = adv_attack.attack(attack_type[attack_type_num], attack_parameter_dict[attack_type[attack_type_num]], benign_normal_testset_dataloader, classifier)
    # adv_all_testset_dataloader = adv_attack.attack(attack_type[attack_type_num], attack_parameter_dict[attack_type[attack_type_num]], benign_attack_testset_dataloader, classifier)

    # attack_dataloader = pre_processing.Merge_Two_Dataloader_To_One(adv_all_testset_dataloader, test_dataloader)
    attack_dataloader = adv_all_testset_dataloader
    pre_processing.test_classifier_on_test(attack_dataloader, classifier)
    # 将测试集数据整体放入generator,恢复数据
    print("防御阶段：二分类器对经生成器还原的干净样本的二分类性能")
    restore_testset_data = generator(attack_dataloader.dataset.data)
    dataset = Data.TensorDataset(restore_testset_data, attack_dataloader.dataset.target)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    restore_testset_dataloader = DataLoader(dataset=dataset, batch_size=pre_processing.batch_size, shuffle=False)
    pre_processing.test_classifier_on_test(restore_testset_dataloader, classifier)

    print("model/generator_" + str(start_time) + ".pth\n\n\n")





