import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pre_processing
from Network_Model import Generator, Discriminator, Classifier
from torch.utils import data as Data

start = time.time()
pre_processing.setup_seed(pre_processing.seed)
writer = SummaryWriter("logs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建网络
generator = Generator().to(device)
discriminator = Discriminator().to(device)
c_g_loss_fn = nn.CrossEntropyLoss().to(device)

# 定义优化器
learning_rate = 1e-4
batch_size = 1024
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), learning_rate)

step = 1
epoch = 50


def Train_Adv_WGAN(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader, adv_normal_trainset_dataloader, adv_attack_trainset_dataloader,a,b,c,d):
    global_step = 0
    for i in range(epoch):
        k = 0
        d_data_length = 0
        g_data_length = 0
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        for _, data in enumerate(zip(benign_normal_trainset_dataloader, benign_attack_trainset_dataloader, adv_attack_trainset_dataloader, adv_normal_trainset_dataloader)):
            k += 1

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
            optimizer_D.zero_grad()

            d_benign_normal_output = discriminator(benign_normal_data).to(device)
            d_benign_attack_output = discriminator(benign_attack_data).to(device)
            d_benign_normal_after_generator_output = discriminator(benign_normal_after_generator).to(device)
            d_benign_attack_after_generator_output = discriminator(benign_attack_after_generator).to(device)
            d_adv_normal_after_generator_output = discriminator(adv_normal_after_generator).to(device)
            d_adv_attack_after_generator_output = discriminator(adv_attack_after_generator).to(device)

            d_g_benign_nomal_loss = torch.mean(d_benign_normal_after_generator_output) + torch.mean(d_benign_normal_output)
            d_g_benign_attack_loss = torch.mean(d_benign_attack_after_generator_output) + torch.mean(d_benign_attack_output)
            d_g_adv_normal_loss = torch.mean(d_adv_normal_after_generator_output) - torch.mean(d_benign_normal_output)
            d_g_adv_attack_loss = torch.mean(d_adv_attack_after_generator_output) - torch.mean(d_benign_attack_output)

            d_total_loss = a * d_g_benign_nomal_loss + b * d_g_benign_attack_loss + c * d_g_adv_normal_loss + d * d_g_adv_attack_loss
            d_total_loss = d_total_loss.to(device)
            d_total_loss.backward(retain_graph=True)
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            if k % step == 0:
                g_data_length += len(data)
                optimizer_G.zero_grad()

                benign_normal_after_generator = generator(benign_normal_data).to(device)
                benign_attack_after_generator = generator(benign_attack_data).to(device)
                adv_normal_after_generator = generator(adv_normal_data).to(device)
                adv_attack_after_generator = generator(adv_attack_data).to(device)

                d_benign_normal_output = discriminator(benign_normal_after_generator).to(device)
                d_benign_attack_output = discriminator(benign_attack_after_generator).to(device)
                d_adv_normal_output = discriminator(adv_normal_after_generator)
                d_adv_attack_output = discriminator(adv_attack_after_generator)

                # c_benign_normal_output = classifier(benign_normal_after_generator).softmax(dim=-1)
                # c_benign_attack_output = classifier(benign_attack_after_generator).softmax(dim=-1)
                # c_g_benign_normal_loss = c_g_loss_fn(c_benign_normal_output, benign_normal_target)
                # c_g_benign_attack_loss = c_g_loss_fn(c_benign_attack_output, benign_attack_target)

                d_g_benign_nomal_loss = torch.mean(d_benign_normal_output)
                d_g_benign_attack_loss = torch.mean(d_benign_attack_output)
                d_g_adv_normal_loss = -torch.mean(d_adv_normal_output)
                d_g_adv_attack_loss = -torch.mean(d_adv_attack_output)


                # g_loss = a * d_g_benign_nomal_loss + b * d_g_benign_attack_loss + c * d_g_adv_attack_loss
                g_loss = a * d_g_benign_nomal_loss + b * d_g_benign_attack_loss + c * d_g_adv_normal_loss + d * d_g_adv_attack_loss
                g_loss = g_loss.to(device)
                g_loss.backward()
                optimizer_G.step()

                with torch.no_grad():
                    d_epoch_loss += d_total_loss
                    g_epoch_loss += g_loss

        with torch.no_grad():
            d_epoch_loss = d_epoch_loss / d_data_length
            g_epoch_loss = g_epoch_loss / g_data_length
            print("epoch: ", i, ", d_epoch_loss", "%.8f" % d_epoch_loss, ", g_epoch_loss", "%.8f" % g_epoch_loss)
            writer.add_scalar("d_epoch_loss", d_epoch_loss, global_step)
            writer.add_scalar("g_epoch_loss", g_epoch_loss, global_step)
            global_step += 1

    return generator

