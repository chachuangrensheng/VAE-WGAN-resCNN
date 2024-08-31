import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import make_grid
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
import os
# 设置环境变量，允许程序继续执行，但可能会有风险
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torch.cuda import empty_cache
from dataloader import CustomImageDataset
from utils import show_and_save,plot_loss,TopHalfCrop

from models4 import VAE_GAN,Discriminator
# nvidia-smi -l 0.2


if __name__=='__main__':
    # 批次大小
    batch_size = 16
    # 数据集路径
    root_dir = './data2'
    # 创建保存模型的文件夹
    models_dir = './models4_1'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 创建保存生成图片的文件夹
    image_dir = './image_mod4_1'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # 定义超参数
    epochs=100
    lr=1e-4
    alpha=0.1 # 定义alpha参数，用于模型调整的权重或比例
    gamma=15  # 定义gamma参数，用于模型中的折扣因子或加权系数

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((150, 100)),  # 根据需要调整图像大小
        TopHalfCrop(),  # 保留上半部分，裁剪掉下半部分
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    train_dataset = CustomImageDataset(root_dir=root_dir, transform=transform, train=True)
    # test_dataset = CustomImageDataset(root_dir=root_dir, transform=transform, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 创建生成模型和鉴别模型
    gen=VAE_GAN().to(device)
    discrim=Discriminator().to(device)
    real_batch = next(iter(train_loader))
    show_and_save("training" ,make_grid((real_batch[0]*0.5+0.5).cpu(),8))

    # 定义损失函数
    criterion=nn.BCELoss().to(device)
    #
    # # 定义RMSprop优化器
    # # optim_E=torch.optim.RMSprop(gen.encoder.parameters(), lr=lr)
    # # optim_D=torch.optim.RMSprop(gen.decoder.parameters(), lr=lr)
    # # optim_Dis=torch.optim.RMSprop(discrim.parameters(), lr=lr*alpha)
    #
    # 定义Adam优化器，并设置动量参数betas，通常第一个值是动量项，第二个值是RMSprop项
    optim_E = torch.optim.Adam(gen.encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(gen.decoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_Dis = torch.optim.Adam(discrim.parameters(), lr=lr * alpha, betas=(0.5, 0.999))

    # 预先准备随机噪声z_fixed和真实样本x_fixed用于后续固定条件下的模型评估
    z_fixed=Variable(torch.randn((batch_size,128)).to(device))
    x_fixed=Variable(real_batch[0].to(device))

    # 开始训练
    prior_loss_list, gan_loss_list, recon_loss_list = [], [], []
    dis_real_list, dis_fake_list, dis_prior_list = [], [], []
    for epoch in range(epochs):


        for i, (data,_) in enumerate(train_loader, 0):
            # 获取批量大小，用于后续的标签和数据准备
            bs=data.size()[0]

            # 创建形状为批量大小的全1张量，用作真实样本的标签
            ones_label=Variable(torch.ones(bs,1).to(device))
            # 创建形状为批量大小的全0张量，用作生成样本的标签
            zeros_label=Variable(torch.zeros(bs,1).to(device))
            # 创建固定形状为64的全0张量，用于特定场景下的标签
            zeros_label1=Variable(torch.zeros(batch_size,1).to(device))
            # 将输入数据转换为变量并移至指定设备（CPU或GPU）
            datav = Variable(data.to(device))
            # 通过生成模型获得输入数据的均值、对数方差和重构输出
            mean, logvar, rec_enc = gen(datav)
            # # 创建形状为64,128的正态分布随机变量，用作生成新样本的输入
            # z_p = Variable(torch.randn(batch_size,128).to(device))
            # # 使用生成模型的解码器生成新样本
            # x_p_tilda = gen.decoder(z_p)

            # 计算真实数据的判别误差
            output = discrim(datav)[0]
            errD_real = criterion(output, ones_label)
            dis_real_list.append(errD_real.item())

            # 计算重构数据的判别误差
            output = discrim(rec_enc)[0]
            errD_rec_enc = criterion(output, zeros_label)
            dis_fake_list.append(errD_rec_enc.item())

            # # 计算带有噪声的重构数据的判别误差
            # output = discrim(x_p_tilda)[0]
            # errD_rec_noise = criterion(output, zeros_label1)
            # dis_prior_list.append(errD_rec_noise.item())

            # 计算总的GAN损失
            gan_loss = errD_real + errD_rec_enc
            gan_loss_list.append(gan_loss.item())

            # 清空判别器优化器的梯度
            optim_Dis.zero_grad()

            # 反向传播GAN损失，保留计算图以便后续计算
            gan_loss.backward(retain_graph=True)

            # 更新判别器的参数
            optim_Dis.step()
            empty_cache()




            # 每2个batch打印一次损失信息
            if i % 2 == 0:
                output = discrim(datav)[0]
                errD_real = criterion(output, ones_label)
                output = discrim(rec_enc)[0]
                errD_rec_enc = criterion(output, zeros_label)
                # output = discrim(x_p_tilda)[0]
                # errD_rec_noise = criterion(output, zeros_label1)
                gan_loss = errD_real + errD_rec_enc
                del output

                # 通过鉴别器获取重构数据的隐藏特征
                x_l_tilda = discrim(rec_enc)[1]
                # 通过鉴别器获取原始数据的隐藏特征
                x_l = discrim(datav)[1]
                # 计算重构损失，即隐藏特征的平方差的均值
                rec_loss = ((x_l_tilda - x_l) ** 2).mean()
                # 计算解码器的错误，结合重构损失和生成对抗损失
                err_dec = rec_loss
                # 将重构损失添加到列表中，用于后续统计或输出
                recon_loss_list.append(err_dec.item())
                # # 清空编码器的优化器的梯度信息
                optim_D.zero_grad()
                # 反向传播计算梯度，保留计算图以便后续计算
                err_dec.backward(retain_graph=True)
                # 更新编码器的参数
                optim_D.step()
                empty_cache()
                # 通过生成模型计算给定数据的均值、对数方差和编码重构值
                mean, logvar, rec_enc = gen(datav)
                # 通过判别模型计算重构编码的判别输出
                x_l_tilda = discrim(rec_enc)[1]
                # 通过判别模型计算原始数据的判别输出
                x_l = discrim(datav)[1]
                # 计算重构损失，即重构数据与原始数据判别输出差的平方的均值
                rec_loss = ((x_l_tilda - x_l) ** 2).mean()
                # 计算先验损失，基于均值、对数方差与先验分布的KL散度
                prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
                # 对先验损失进行归一化处理，得到每个数据元素的平均损失
                prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
                # 将先验损失的值添加到列表中，用于后续处理
                prior_loss_list.append(prior_loss.item())
                # 计算编码器的错误项，即先验损失与重构损失的加权和
                err_enc = prior_loss + 5 * rec_loss

                optim_E.zero_grad()
                err_enc.backward(retain_graph=True)
                optim_E.step()
                print('[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f'
                      % (epoch,epochs, i, len(train_loader),
                         gan_loss.item(), prior_loss.item(),rec_loss.item(),errD_real.item(),errD_rec_enc.item()))

        # # 每个epoch分别保存编码器、解码器和判别器
        # encoder_path = os.path.join(models_dir, 'encoder_epoch_{}.pth'.format(epoch))
        # decoder_path = os.path.join(models_dir, 'decoder_epoch_{}.pth'.format(epoch))
        # discriminator_path = os.path.join(models_dir, 'discriminator_epoch_{}.pth'.format(epoch))
        #
        # torch.save(gen.encoder.state_dict(), encoder_path)
        # torch.save(gen.decoder.state_dict(), decoder_path)
        # torch.save(gen.discriminator.state_dict(), discriminator_path)

        # 每个epoch保存一次整个VAE_GAN模型
        vae_gan_path = os.path.join(models_dir, 'vae_gan_epoch_{}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': gen.encoder.state_dict(),
            'decoder_state_dict': gen.decoder.state_dict(),
            'discriminator_state_dict': gen.discriminator.state_dict(),
        }, vae_gan_path)

        # 从生成模型中获取特定输出，用于后续操作
        b = gen(x_fixed)[2]
        # 从计算图中分离张量，避免梯度计算
        b = b.detach()
        # 使用固定的随机向量通过生成模型的解码器，获取特定输出
        c = gen.decoder(z_fixed)
        # 从计算图中分离张量，避免梯度计算
        c = c.detach()
        # 使用make_grid函数将特定输出转换为网格形式，并保存为图片
        show_and_save(f'{image_dir}/train_noise_epoch_{epoch}', make_grid((c * 0.5 + 0.5).cpu(), 8))
        show_and_save(f'{image_dir}/train_epoch_{epoch}', make_grid((b*0.5+0.5).cpu(),8))

    # 训练结束后保存最后的模型状态
    print('Saving the final model')
    # 分别保存编码器、解码器和判别器最后的模型状态
    encoder_final_path = os.path.join(models_dir, 'encoder_final.pth')
    decoder_final_path = os.path.join(models_dir, 'decoder_final.pth')
    discriminator_final_path = os.path.join(models_dir, 'discriminator_final.pth')

    torch.save(gen.encoder.state_dict(), encoder_final_path)
    torch.save(gen.decoder.state_dict(), decoder_final_path)
    torch.save(gen.discriminator.state_dict(), discriminator_final_path)

    # 保存整个VAE_GAN模型的最终状态
    vae_gan_final_path = os.path.join(models_dir, 'vae_gan_final.pth')
    torch.save({
        'encoder_state_dict': gen.encoder.state_dict(),
        'decoder_state_dict': gen.decoder.state_dict(),
        'discriminator_state_dict': gen.discriminator.state_dict(),
    }, vae_gan_final_path)

    # 绘制损失曲线
    plot_loss(prior_loss_list, 'models4_1_prior_loss0.png')  # 先验损失误差
    plot_loss(recon_loss_list, 'models4_1_recon_loss0.png')  # 重建损失误差
    plot_loss(gan_loss_list, 'models4_1_gan_loss0.png')   # GAN损失误差

