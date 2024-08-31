import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
import os
import time
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
from dataloader import CustomImageDataset
from models import VAE_GAN,Discriminator
from utils import show_and_save,plot_loss,TopHalfCrop

if __name__=='__main__':
    # 批次大小
    batch_size = 1
    class_num = 603
    # 数据集路径
    root_dir = './data1'
    # 定义gamma参数，用于模型中的折扣因子或加权系数
    gamma=15

    # # 把保存在txt文件中的阈值读取出来
    # with open('eta.txt', 'r') as f:
    #     eta = float(f.read())
    #     print("读取到的阈值:", eta)

    # 打开文件并读取内容
    with open('eta1.txt', 'r') as f:
        content = f.read()
    # 去除可能的空白字符，并按逗号分割字符串
    eta_values = content.strip().split(',')
    # 分别将分割后的字符串转换为浮点数或整数，并赋值给新变量
    # 这里假设阈值是浮点数，如果它们是整数，使用 int() 替换 float()
    eta_l = float(eta_values[0].strip())
    eta_h = float(eta_values[1].strip())
    # 打印读取的值以验证
    print("读取的阈值下限:", eta_l)
    print("读取的阈值上限:", eta_h)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((150, 100)),  # 根据需要调整图像大小
        TopHalfCrop(),  # 保留上半部分，裁剪掉下半部分
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    test_dataset = CustomImageDataset(root_dir=root_dir, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    real_batch = next(iter(test_loader))



    # 初始化模型结构
    # 创建VAE_GAN实例，以及单独的编码器、解码器和判别器实例
    vae_gan = VAE_GAN().to(device)
    # encoder = vae_gan.encoder
    # decoder = vae_gan.decoder
    # discriminator = vae_gan.discriminator


    # 加载保存的状态字典
    # 请确保提供的路径是正确的，并且对应于您要加载的模型
    models_dir = 'models0'  # 模型保存的文件夹
    encoder_final_path = os.path.join(models_dir, 'encoder_final.pth')
    decoder_final_path = os.path.join(models_dir, 'decoder_final.pth')
    discriminator_final_path = os.path.join(models_dir, 'discriminator_final.pth')

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

    # 预先准备随机噪声z_fixed和真实样本x_fixed用于后续固定条件下的模型测试
    z_fixed = Variable(torch.randn((batch_size, 128)).to(device))
    x_fixed = Variable(real_batch[0].to(device))
    # 从生成模型中获取特定输出，用于后续操作
    b = vae_gan(x_fixed)[2]
    # 从计算图中分离张量，避免梯度计算
    b = b.detach()
    # 使用固定的随机向量通过生成模型的解码器，获取特定输出
    c = vae_gan.decoder(z_fixed)
    # 从计算图中分离张量，避免梯度计算
    c = c.detach()
    # 使用make_grid函数将特定输出转换为网格形式，并保存为图片
    show_and_save(f'test_noise', make_grid((c * 0.5 + 0.5).cpu(), 1))
    show_and_save(f'test', make_grid((b * 0.5 + 0.5).cpu(), 1))

    prior_loss_list, gan_loss_list, recon_loss_list = [], [], []
    dis_real_list, dis_fake_list, dis_prior_list = [], [], []
    # 预测标签
    y = []
    # 真实标签
    all_labels = []
    # 正确率
    correct = 0
    # 计数器
    counter = 0
    y_batch = []
    with torch.no_grad():
        for i, (data,lable) in enumerate(test_loader, 0):
            counter += 1
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
            lable = Variable(lable.to(device))
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
            rec_loss = ((x_l_tilda - x_l) ** 2)
            # # 计算解码器的错误，结合重构损失和生成对抗损失
            # err_dec = gamma * rec_loss - gan_loss
            # 将重构损失添加到列表中，用于后续统计或输出
            recon_loss_list.append(rec_loss.mean().item())


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
            scores = rec_loss.mean()
            scores = np.array(scores.cpu())  # 首先将张量复制到CPU，然后转换为NumPy数组
            # from sklearn.preprocessing import MinMaxScaler
            #
            # # 假设 normal_scores 是一个一维数组或列表
            # scaler = MinMaxScaler(feature_range=(0, 1))
            # scores_normalized = scaler.fit_transform(scores.reshape(-1, 1))


            # Compare the scores with η to determine anomalies
            anomalies = (scores < eta_l) or (scores > eta_h)
            if anomalies:
                a = 0
                y.append(a)
                y_batch.append(a)
                # print("异常!")
            else:
                a = 1
                y.append(a)
                y_batch.append(a)
                # print("正常")
            all_labels.extend(lable.cpu().numpy())
            a_tensor = torch.tensor(a)
            # lable_tensor = torch.tensor(lable)
            correct += (a_tensor.float() == lable.float()).sum().item()
            if counter == class_num:
                counter = 0
                y_batch = sum(y_batch)
                if y_batch / class_num >= 0.9:
                    print("正常")
                else:
                    print("异常！")
                y_batch = []


    # 计算准确率
    y = np.array(y)
    accuracy = 100 * correct / len(all_labels)
    print(f'Accuracy of the model on the test images: {accuracy}%')

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(all_labels, y)
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=[str(i) for i in range(2)],
                                  index=[str(i) for i in range(2)])

    # 计算正确率和错误率矩阵
    accuracy_matrix = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    error_matrix = 1 - accuracy_matrix

    # 将正确率和错误率转换为一维数组
    accuracy_array = accuracy_matrix.ravel()
    error_array = error_matrix.ravel()

    # 创建正确率和错误率的DataFrame
    accuracy_matrix_df = pd.DataFrame([accuracy_array], columns=conf_matrix_df.columns)
    error_matrix_df = pd.DataFrame([error_array], columns=conf_matrix_df.columns)

    # 绘制混淆矩阵和正确率/错误率矩阵
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 1, 1], wspace=0.3, hspace=0.3)

    # 混淆矩阵
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax0)
    ax0.set_xlabel('Predicted')
    ax0.set_ylabel('True')

    # 正确率矩阵
    ax1 = fig.add_subplot(gs[0, 1])
    sns.heatmap(pd.DataFrame(accuracy_matrix), annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=False, yticklabels=False)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_title('Accuracy')

    # 错误率矩阵
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(pd.DataFrame(error_matrix), annot=True, fmt='.2%', cmap='Reds', cbar=False, ax=ax2,
                xticklabels=False, yticklabels=False)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    ax2.set_title('Error')

    # 保存混淆矩阵图形
    plt.savefig('test_confusion_matrix.png')
    plt.show()


