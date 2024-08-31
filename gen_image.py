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
from dataloader import CustomImageDataset
from models4 import VAE_GAN,Discriminator
from utils import show_and_save,plot_loss,TopHalfCrop

if __name__=='__main__':
    # 批次大小
    batch_size = 16
    # 数据集路径
    root_dir = './data2'
    # 加载保存的状态字典
    models_dir = 'models4_1'  # 模型保存的文件夹
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

    # 初始化模型结构
    # 创建VAE_GAN实例
    vae_gan = VAE_GAN().to(device)


    encoder_final_path = os.path.join(models_dir, 'encoder_final.pth')
    decoder_final_path = os.path.join(models_dir, 'decoder_final.pth')
    discriminator_final_path = os.path.join(models_dir, 'discriminator_final.pth')

    # 加载整个模型
    vae_gan_final_path = os.path.join(models_dir, 'vae_gan_final.pth')
    vae_gan_checkpoint = torch.load(vae_gan_final_path, map_location=device)
    vae_gan.encoder.load_state_dict(vae_gan_checkpoint['encoder_state_dict'])
    vae_gan.decoder.load_state_dict(vae_gan_checkpoint['decoder_state_dict'])
    vae_gan.discriminator.load_state_dict(vae_gan_checkpoint['discriminator_state_dict'])
    print('load model success')
    # 确保在加载模型后将其移动到GPU上
    vae_gan = vae_gan.to(device)

    # 将模型设置为评估模式
    vae_gan.eval()

    # 预先准备随机噪声z_fixed和真实样本x_fixed用于后续固定条件下的模型测试
    z_fixed = Variable(torch.randn((batch_size, 128)).to(device))
    x_fixed = Variable(real_batch[0].to(device))
    # 从生成模型中获取特定输出，用于后续操作
    b = vae_gan(x_fixed)[2]
    # 从计算图中分离张量，避免梯度计算
    b = b.detach()
    # # 使用固定的随机向量通过生成模型的解码器，获取特定输出
    # c = vae_gan.decoder(z_fixed)
    # # 从计算图中分离张量，避免梯度计算
    # c = c.detach()
    # # 使用make_grid函数将特定输出转换为网格形式，并保存为图片
    # show_and_save(f'test_noise', make_grid((c * 0.5 + 0.5).cpu(), 8))

    show_and_save("testing3", make_grid((real_batch[0] * 0.5 + 0.5).cpu(), 4))
    show_and_save(f'gen_text3', make_grid((b * 0.5 + 0.5).cpu(), 4))