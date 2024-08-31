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
from sklearn.decomposition import PCA
from dataloader import CustomImageDataset
from utils import show_and_save,plot_loss,TopHalfCrop

from models import VAE_GAN,Discriminator


if __name__=='__main__':
    # 批次大小
    batch_size = 1
    # 数据集路径
    root_dir = './data1'
    # 模型保存的文件夹
    models_dir = 'models'
    # 定义gamma参数，用于模型中的折扣因子或加权系数
    gamma=15
    # 阈值调整系数
    C = 3
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((150, 100)),  # 根据需要调整图像大小
        TopHalfCrop(),  # 保留上半部分，裁剪掉下半部分
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    test_dataset = CustomImageDataset(root_dir=root_dir, transform=transform, train=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    real_batch = next(iter(test_loader))
    show_and_save("testing", make_grid((real_batch[0] * 0.5 + 0.5).cpu(), 1))


    # 初始化模型结构
    # 创建VAE_GAN实例，以及单独的编码器、解码器和判别器实例
    vae_gan = VAE_GAN().to(device)
    # encoder = vae_gan.encoder
    # decoder = vae_gan.decoder
    # discriminator = vae_gan.discriminator


    # 加载保存的状态字典
    # 请确保提供的路径是正确的，并且对应于您要加载的模型
    # encoder_final_path = os.path.join(models_dir, 'encoder_final.pth')
    # decoder_final_path = os.path.join(models_dir, 'decoder_final.pth')
    # discriminator_final_path = os.path.join(models_dir, 'discriminator_final.pth')

    # # 加载单独的模型组件
    # encoder_checkpoint = torch.load(encoder_final_path, map_location=device)
    # decoder_checkpoint = torch.load(decoder_final_path, map_location=device)
    # discriminator_checkpoint = torch.load(discriminator_final_path, map_location=device)
    #
    # # 将状态字典应用到模型
    # encoder.load_state_dict(encoder_checkpoint)
    # decoder.load_state_dict(decoder_checkpoint)
    # discriminator.load_state_dict(discriminator_checkpoint)

    # 加载整个模型
    vae_gan_final_path = os.path.join(models_dir, 'vae_gan_final.pth')
    vae_gan_checkpoint = torch.load(vae_gan_final_path, map_location=device)
    vae_gan.encoder.load_state_dict(vae_gan_checkpoint['encoder_state_dict'])
    vae_gan.decoder.load_state_dict(vae_gan_checkpoint['decoder_state_dict'])
    vae_gan.discriminator.load_state_dict(vae_gan_checkpoint['discriminator_state_dict'])
    print('load model success')
    # 确保在加载模型后将其移动到GPU上
    vae_gan = vae_gan.to(device)
    discrim = Discriminator().to(device)

    # 定义损失函数
    criterion = nn.BCELoss().to(device)
    # 将模型设置为评估模式
    vae_gan.eval()
    discrim.eval()

    # # 预先准备随机噪声z_fixed和真实样本x_fixed用于后续固定条件下的模型测试
    # z_fixed = Variable(torch.randn((batch_size, 128)).to(device))
    # x_fixed = Variable(real_batch[0].to(device))
    # # 从生成模型中获取特定输出，用于后续操作
    # b = vae_gan(x_fixed)[2]
    # # 从计算图中分离张量，避免梯度计算
    # b = b.detach()
    # # 使用固定的随机向量通过生成模型的解码器，获取特定输出
    # c = vae_gan.decoder(z_fixed)
    # # 从计算图中分离张量，避免梯度计算
    # c = c.detach()
    # # 使用make_grid函数将特定输出转换为网格形式，并保存为图片
    # show_and_save(f'test_noise', make_grid((c * 0.5 + 0.5).cpu(), 8))
    # show_and_save(f'test', make_grid((b * 0.5 + 0.5).cpu(), 8))

    prior_loss_list, gan_loss_list, recon_loss_list = [], [], []
    dis_real_list, dis_fake_list, dis_prior_list = [], [], []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader, 0):
            # 获取批量大小，用于后续的标签和数据准备
            bs = data.size()[0]

            # # 创建形状为批量大小的全1张量，用作真实样本的标签
            # ones_label = Variable(torch.ones(bs, 1).to(device))
            # # 创建形状为批量大小的全0张量，用作生成样本的标签
            # zeros_label = Variable(torch.zeros(bs, 1).to(device))
            # # 创建固定形状为64的全0张量，用于特定场景下的标签
            # zeros_label1 = Variable(torch.zeros(batch_size, 1).to(device))
            # 将输入数据转换为变量并移至指定设备（CPU或GPU）
            datav = Variable(data.to(device))
            # 通过生成模型获得输入数据的均值、对数方差和重构输出
            mean, logvar, rec_enc = vae_gan(datav)
            # # 创建形状为64,128的正态分布随机变量，用作生成新样本的输入
            # z_p = Variable(torch.randn(batch_size, 128).to(device))
            # # 使用生成模型的解码器生成新样本
            # x_p_tilda = vae_gan.decoder(z_p)

            # # 计算真实数据的判别误差
            # output = discrim(datav)[0]
            # errD_real = criterion(output, ones_label)
            # dis_real_list.append(errD_real.item())
            #
            # # 计算重构数据的判别误差
            # output = discrim(rec_enc)[0]
            # errD_rec_enc = criterion(output, zeros_label)
            # dis_fake_list.append(errD_rec_enc.item())
            #
            # # 计算带有噪声的重构数据的判别误差
            # output = discrim(x_p_tilda)[0]
            # errD_rec_noise = criterion(output, zeros_label1)
            # dis_prior_list.append(errD_rec_noise.item())
            #
            # # 计算总的GAN损失
            # gan_loss = errD_real + errD_rec_enc + errD_rec_noise
            # gan_loss_list.append(gan_loss.item())
            #
            #
            # output = discrim(datav)[0]
            # errD_real = criterion(output, ones_label)
            # output = discrim(rec_enc)[0]
            # errD_rec_enc = criterion(output, zeros_label)
            # output = discrim(x_p_tilda)[0]
            # errD_rec_noise = criterion(output, zeros_label1)
            # gan_loss = errD_real + errD_rec_enc + errD_rec_noise


            # 通过鉴别器获取重构数据的隐藏特征
            x_l_tilda = discrim(rec_enc)[1]
            # 通过鉴别器获取原始数据的隐藏特征
            x_l = discrim(datav)[1]
            # 计算重构损失，即隐藏特征的平方差的均值
            # rec_loss = ((x_l_tilda - x_l) ** 2).mean()
            rec_loss = ((x_l_tilda - x_l) ** 2)
            # # 计算解码器的错误，结合重构损失和生成对抗损失
            # err_dec = gamma * rec_loss - gan_loss
            # 将重构损失添加到列表中，用于后续统计或输出
            # recon_loss_list.append(rec_loss.item())
            # recon_loss_list.extend(rec_loss.cpu().numpy())
            recon_loss_list.extend(rec_loss.cpu().numpy())

        # # 通过生成模型计算给定数据的均值、对数方差和编码重构值
        # mean, logvar, rec_enc = vae_gan(datav)
        # # 通过判别模型计算重构编码的判别输出
        # x_l_tilda = discrim(rec_enc)[1]
        # # 通过判别模型计算原始数据的判别输出
        # x_l = discrim(datav)[1]
        # # 计算重构损失，即重构数据与原始数据判别输出差的平方的均值
        # rec_loss = ((x_l_tilda - x_l) ** 2).mean()
        # # 计算先验损失，基于均值、对数方差与先验分布的KL散度
        # prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
        # # 对先验损失进行归一化处理，得到每个数据元素的平均损失
        # prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
        # # 将先验损失的值添加到列表中，用于后续处理
        # prior_loss_list.append(prior_loss.item())
        # # 计算编码器的错误项，即先验损失与重构损失的加权和
        # err_enc = prior_loss + 5 * rec_loss
    # Calculate the anomaly score
    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=1)
    recon_loss_list = pca.fit_transform(recon_loss_list)
    normal_scores = recon_loss_list

    # normal_scores = np.array(normal_scores)  # 转换为 NumPy 数组
    # from sklearn.preprocessing import MinMaxScaler
    # # 假设 normal_scores 是一个一维数组或列表
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # normal_scores_normalized = scaler.fit_transform(normal_scores.reshape(-1, 1))

    mean_score = np.mean(normal_scores)
    std_score = np.std(normal_scores)

    eta_l = mean_score - C * std_score
    eta_h = mean_score + C * std_score

    print("计算得到的阈值下限:", eta_l, "计算得到的阈值上限:", eta_h)
    # 把eta_mean保存到txt文件中
    with open('eta.txt', 'w') as f:
        f.write(str(eta_l) + ", " + str(eta_h))


