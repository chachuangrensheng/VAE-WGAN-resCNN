import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
height = 99
width = 99


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
fc_height = 99
fc_width = 99


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=51, stride=2)  # in_channels=3
        self.bn1 = nn.BatchNorm2d(8, affine=True)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=51, stride=2)
        self.bn2 = nn.BatchNorm2d(16, affine=True)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=51, stride=2)
        self.bn3 = nn.BatchNorm2d(32, affine=True)
        self.fc_mean = nn.Linear(32 * fc_height * fc_width, 128)
        self.fc_logvar = nn.Linear(32 * fc_height * fc_width, 128)  # latent dim=128

    def forward(self, x):
        batch_size = x.size()[0]
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out = out.view(batch_size, -1)
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(128, fc_height * fc_width * 32)
        self.bn1 = nn.BatchNorm1d(fc_height * fc_width * 32, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(32, 32, 5, stride=2, padding=51)
        self.bn2 = nn.BatchNorm2d(32, affine=True)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 5, stride=2, padding=51)
        self.bn3 = nn.BatchNorm2d(16, affine=True)
        self.deconv3 = nn.ConvTranspose2d(16, 8, 5, stride=2, padding=51)
        self.bn4 = nn.BatchNorm2d(8, affine=True)
        self.deconv4 = nn.ConvTranspose2d(8, 3, 5, stride=1, padding=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 32, fc_height, fc_width)
        x = self.relu(self.bn2(self.deconv1(x)))
        x = self.relu(self.bn3(self.deconv2(x)))
        x = self.relu(self.bn4(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2, stride=1)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(6, 12, 5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 24, 5, padding=2, stride=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 48, 5, padding=2, stride=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(fc_height * fc_width * 48, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn3(self.conv4(x)))
        x = x.view(-1, 48 * fc_height * fc_width)
        x1 = self.fc1(x)
        x = self.sigmoid(x1)

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


