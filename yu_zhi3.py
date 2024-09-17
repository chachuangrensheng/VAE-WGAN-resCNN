import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from torch.autograd import Variable
from torchvision.utils import make_grid
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
import os
from sklearn.decomposition import PCA
from dataloader import CustomImageDataset
from utils import show_and_save,plot_loss,TopHalfCrop

from resmodels1 import VAE_GAN,Discriminator
# from models4_5 import VAE_GAN,Discriminator


if __name__=='__main__':
    # 批次大小
    batch_size = 1
    # 每包数据大小
    pack_size = 600
    # 故障数据大小
    fault_size = 100
    # 数据集路径
    root_dir = './data2'
    # 模型保存的文件夹
    models_dir = 'resmodels1'
    # 定义gamma参数，用于模型中的折扣因子或加权系数
    gamma=15
    # 阈值调整系数
    C = 3
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((150, 99)),  # 根据需要调整图像大小
        TopHalfCrop(),  # 保留上半部分，裁剪掉下半部分
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    test_dataset = CustomImageDataset(root_dir=root_dir, transform=transform, train=False, yu=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    real_batch = next(iter(test_loader))
    show_and_save("testing", make_grid((real_batch[0] * 0.5 + 0.5).cpu(), 1))

    # 定义损失函数
    criterion=nn.BCELoss().to(device)

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
        # 设置随机种子
        torch.manual_seed(0)
        # 随机数生成的库的种子设置
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
        for i, (data, _) in enumerate(test_loader, 0):
            # 获取批量大小，用于后续的标签和数据准备
            bs = data.size()[0]
            datav = Variable(data.to(device))
            # 通过生成模型获得输入数据的均值、对数方差和重构输出
            # 设置随机种子
            torch.manual_seed(0)
            mean, logvar, rec_enc = vae_gan(datav)
            # rec_enc, z = vae_gan(datav)
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
            # 创建形状为批量大小的全1张量，用作真实样本的标签
            # ones_label = Variable(torch.ones(bs, 1).to(device))
            # # 创建形状为批量大小的全0张量，用作生成样本的标签
            # zeros_label = Variable(torch.zeros(bs, 1).to(device))
            # # 创建固定形状为64的全0张量，用于特定场景下的标签
            # zeros_label1 = Variable(torch.zeros(batch_size, 1).to(device))
            # # 创建形状为64,128的正态分布随机变量，用作生成新样本的输入
            # z_p = Variable(torch.randn(batch_size, 128).to(device))
            # # 使用生成模型的解码器生成新样本
            # # 设置随机种子
            # torch.manual_seed(0)
            # x_p_tilda = vae_gan.decoder(z_p)

            # output, x_l = discrim(datav)
            # # errD_real = criterion(output, ones_label)
            # output, x_l_tilda = discrim(rec_enc)
            # errD_rec_enc = criterion(output, zeros_label)
            # output = discrim(x_p_tilda)[0]
            # errD_rec_noise = criterion(output, zeros_label1)
            # gan_loss = errD_real + errD_rec_enc
            # gan_loss_list.append(gan_loss.cpu().numpy())

            # 计算重构损失，即隐藏特征的平方差的均值
            # rec_loss = ((x_l_tilda - x_l) ** 2).mean()
            # rec_loss = ((x_l_tilda - x_l) ** 2)
            # rec_loss_mean = rec_loss.mean()
            # 计算解码器的错误，结合重构损失和生成对抗损失
            # err_dec = rec_loss_mean
            # 通过鉴别器获取重构数据的隐藏特征
            # x_l_tilda = discrim(rec_enc)[1]
            # 通过鉴别器获取原始数据的隐藏特征
            # x_l = discrim(datav)[1]
            # rec_loss = ((x_l_tilda + x_l) ** 2)
            # err_dec = x_l.mean()
            # err_dec = (rec_enc - datav).mean()
            err_dec = rec_enc.mean()
            # err_dec = rec_loss.mean()
            # 将重构损失添加到列表中，用于后续统计或输出
            recon_loss_list.append(err_dec.cpu().numpy())
            # recon_loss_list.append(rec_loss_mean.cpu().numpy())
            # recon_loss_list.extend(rec_loss.cpu().numpy())

    # 将列表转换为 NumPy 数组以便进行数值操作
    recon_loss_list = np.array(recon_loss_list)
    # # 过滤数据
    # filtered_scores_list = []
    # for i in range(int(len(recon_loss_list) / fault_size)):
    #     recon_loss_score = recon_loss_list[i * fault_size:(i + 1) * fault_size]
    #     # 计算均值和标准差
    #     mean = np.mean(recon_loss_score)
    #     std = np.std(recon_loss_score)
    #
    #     # 定义剔除异常值阈值使用3倍标准差
    #     threshold = 3
    #     lower_bound = mean - threshold * std
    #     upper_bound = mean + threshold * std
    #
    #     # 过滤数据
    #     for j in range(len(recon_loss_score)):
    #         if recon_loss_score[j] < lower_bound:
    #             recon_loss_score[j] = lower_bound
    #         elif recon_loss_score[j] > upper_bound:
    #             recon_loss_score[j] = upper_bound
    #     filtered_scores_list.append(recon_loss_score)
    # Calculate the anomaly score
    # 最大最小归一化
    # 计算最小值和最大值
    filtered_scores_list = np.array(recon_loss_list)
    filtered_scores_list = filtered_scores_list.reshape(-1,1)
    min_value = np.min(filtered_scores_list)
    max_value = np.max(filtered_scores_list)
    # 应用最小-最大标准化
    # filtered_scores_list = (filtered_scores_list - min_value) / (max_value - min_value)
    # 将[0, 1]范围的数据缩放到[-1, 1]的范围
    # filtered_scores_list = filtered_scores_list * 2 - 1

    plt.figure(figsize=(10, 5))
    for i in range(int(len(filtered_scores_list) / pack_size)):
        recon_loss_score = filtered_scores_list[i * pack_size:(i + 1) * pack_size]
        # mean_recon_loss_score = np.mean(recon_loss_score)
        # std_recon_loss_score = np.std(recon_loss_score)
        # print("mean_recon_loss_list:", mean_recon_loss_score)
        # print("std_recon_loss_list:", std_recon_loss_score)

        # 绘制recon_loss_list的折线图
        # recon_loss_array = np.array(recon_loss_score)
        # recon_loss_array = recon_loss_array.reshape(-1,1)
        # plt.figure(figsize=(10, 5))
        # plt.plot(recon_loss_array, label="Reconstruction Loss")
        # plt.grid(True)  # 可以添加网格线
        # name = str(i)+"Reconstruction Loss"
        # plt.title(name)
        # plt.savefig(name)  # 保存图像
        # plt.show()

        # Apply PCA to reduce the dimensionality
        # pca = PCA(n_components=1)
        # recon_loss_score = pca.fit_transform(recon_loss_score)


        pca_scores = recon_loss_score
        name = f"{i}scores"
        plt.plot(pca_scores, label=name)

        # 计算均值和标准差
        if i == 0:
            pca_scores = recon_loss_score
            # 绘制直方图
            # # 返回的bins为列表，为频率分区
            # n, bins, patches = plt.hist(pca_scores,
            #                             density=True,  # true表示绘制的频率直方图总面积为1
            #                             bins=30)  # 默认分为10组，可以通过调节bins值，扩大分组。
            #
            # # 计算样本的均值和标准差
            # mu = pca_scores.mean()
            # sigma = pca_scores.std()
            # # 拟合一条最佳正态分布曲线y，y的值和分区bins的值对应
            # y = norm.pdf(bins, mu, sigma)
            # title = " mu = {:.2f},  std = {:.2f}".format(mu, sigma)
            # plt.title(title, fontsize=14)
            # plt.plot(bins, y, 'r--')  # 绘制y的曲线
            # # 添加轴标签
            # plt.xlabel('Refactoring features', fontsize=14)  # 横轴标签
            # plt.ylabel('Frequency', fontsize=14)  # 纵轴标签


            # Q-Q图
            # from scipy import stats
            # pca_scores = np.ravel(pca_scores)
            # # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            # stats.probplot(pca_scores, dist=stats.norm, plot=plt)
            # plt.xlabel('Expected normal value', fontsize=14)
            # plt.ylabel('Original data value', fontsize=14)
            # plt.xlim(0, 0.15)
            # plt.title(name)

            # name = f"{i}scores"
            # plt.plot(pca_scores, label=name)
            mean_score = np.mean(pca_scores)
            std_score = np.std(pca_scores)
            print("mean_score:", mean_score)
            print("std_score:", std_score)
            eta_l = mean_score - C * std_score
            eta_h = mean_score + C * std_score
            # 添加水平标线
            plt.axhline(y=eta_l, color='r', linestyle='--', label=f'Lowest_eta: {eta_l:.4f}')
            plt.axhline(y=eta_h, color='g', linestyle='--', label=f'Highest_eta: {eta_h:.4f}')
            print("计算得到的阈值下限:", eta_l, "计算得到的阈值上限:", eta_h)
            # 把eta_mean保存到txt文件中
            with open('eta.txt', 'w') as f:
                f.write(str(eta_l) + ", " + str(eta_h))


    plt.legend()
    plt.grid(True)  # 可以添加网格线
    plt.title("mean_scores")
    plt.savefig("mean_scores")  # 保存图像
    # plt.savefig("Q-Q_scores")  # 保存图像
    plt.show()


