import torch
import torch.nn as nn
from torch.autograd import Variable
from cbam import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
height = 75
width = 100


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 计算特征图的尺寸函数
def get_feature_size(input_size):
    feature_size = input_size
    for i in range(3):
        feature_size = (feature_size - 5 + 2 * 2) // 2 + 1
    return feature_size


# 计算特征图的尺寸，125x187经过三次stride=2的卷积操作后的尺寸为16x24
# fc_height = get_feature_size(height)
# fc_width = get_feature_size(width)
fc_height = 10
fc_width = 13
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, stride=2):
        super(ResidualBlock, self).__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.relu = nn.LeakyReLU(0.2)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=0.9)

    def forward(self, x):
        identity = self.downsample(x)
        identity = self.bn3(identity)

        out = self.conv2(x)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, stride=2):
        super(ResidualBlock1, self).__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.relu = nn.LeakyReLU(0.2)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=0.9)

    def forward(self, x):
        identity = self.downsample(x)
        identity = self.bn3(identity)

        out = self.conv2(x)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, stride=2):
        super(ResidualBlockUp, self).__init__()
        self.deconv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.relu = nn.LeakyReLU(0.2)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=0.9)

    def forward(self, x):
        identity = self.upsample(x)
        identity = self.bn3(identity)

        out = self.deconv2(x)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
# 残差网络类，用于Decoder的上采样
class ResidualBlockUp1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, stride=2):
        super(ResidualBlockUp1, self).__init__()
        # self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.deconv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.relu = nn.LeakyReLU(0.2)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9)
        # self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride)
        # self.bn3 = nn.BatchNorm2d(out_channels, momentum=0.9)

    def forward(self, x):
        # identity = self.upsample(x)
        # identity = self.bn3(identity)

        # out = self.deconv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.deconv2(x)
        out = self.bn2(out)

        # out += identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            ResidualBlock(3, 64, 5, padding=2, stride=2),
            # CBAMBlock(channel=64, kernel_size=7)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, 5, padding=2, stride=2),
            # CBAMBlock(channel=128, kernel_size=5)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, 5, padding=2, stride=2),
            # CBAMBlock(channel=256, kernel_size=3)
        )
        self.relu = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(256 * fc_height * fc_width, 2048)
        self.bn4 = nn.BatchNorm1d(2048, momentum=0.9)
        self.fc_mean = nn.Linear(2048, 128)
        self.fc_logvar = nn.Linear(2048, 128)  # latent dim=128

    def forward(self, x):
        batch_size = x.size()[0]
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(batch_size, -1)
        out = self.relu(self.bn4(self.fc1(out)))
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(128, fc_height * fc_width * 256)
        self.bn1 = nn.BatchNorm1d(fc_height * fc_width * 256, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.layer1 = nn.Sequential(
            ResidualBlockUp1(256, 256, 5, padding=2, stride=2),
            # CBAMBlock(channel=256, kernel_size=3)
        )
        self.layer2 = nn.Sequential(
            ResidualBlockUp1(256, 128, 5, padding=2, stride=2),
            # CBAMBlock(channel=128, kernel_size=5)
        )
        self.layer3 = nn.Sequential(
            ResidualBlockUp1(128, 64, 5, padding=2, stride=2),
            # CBAMBlock(channel=64, kernel_size=7)
        )
        self.deconv4 = nn.ConvTranspose2d(64, 3, 5, stride=1, padding=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 256, fc_height, fc_width)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.tanh(self.deconv4(x))
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2, stride=1)
        self.relu = nn.LeakyReLU(0.2)
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 128, 5, padding=2, stride=2),
            # CBAMBlock(channel=128, kernel_size=7)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(128, 256, 5, padding=2, stride=2),
            # CBAMBlock(channel=256, kernel_size=5)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(256, 256, 5, padding=2, stride=2),
            # CBAMBlock(channel=256, kernel_size=3)
        )
        self.fc1 = nn.Linear(fc_height * fc_width * 256, 512)
        self.bn4 = nn.BatchNorm1d(512, momentum=0.9)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 256 * fc_height * fc_width)
        x1 = x
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x, x1



class VAE_GAN(nn.Module):
    def __init__(self):
        super(VAE_GAN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

    def forward(self, x):
        bs = x.size()[0]
        z_mean, z_logvar = self.encoder(x)
        std = z_logvar.mul(0.5).exp_()

        # sampling epsilon from normal distribution
        epsilon = Variable(torch.randn(bs, 128).to(device))
        z = z_mean + std * epsilon
        x_tilda = self.decoder(z)

        return z_mean, z_logvar, x_tilda


