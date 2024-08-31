import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision
# import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
from PIL import Image
# def dataloader(batch_size):
#     dataroot="./data0"
#     transform=transforms.Compose([ transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
#     dataset=torchvision.datasets.MNIST(root=dataroot, train=True,transform=transform, download=True)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return data_loader
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, yu=False):
        # 存储所有图像的路径和标签
        self.images = []
        self.labels = []
        self.transform = transform
        self.root_dir = root_dir
        self.label_to_idx = {label: idx for idx, label in enumerate(os.listdir(root_dir))}
        self.train = train

        for label in os.listdir(self.root_dir):
            num_file_to_read = len(os.listdir(self.root_dir)) - 1
            folder_path = os.path.join(self.root_dir, label)
            if os.path.isdir(folder_path):
                # 遍历文件夹内的图像文件
                if label == '0'and (self.train or yu):
                    i = 0
                    for img_file in os.listdir(folder_path):
                        # 只读取每个folder_path中的前 600 张图片
                        i = i + 1
                        if i > 600 :
                            break
                        img_path = os.path.join(folder_path, img_file)
                        if os.path.isfile(img_path):
                            self.images.append(img_path)
                            # 将文件夹名称作为标签
                            self.labels.append(1)
                elif  not self.train:
                    if label == '0'and not yu:
                        i = 600
                        for img_file in os.listdir(folder_path):
                            # 跳过前600个文件
                            i = i - 1
                            if i >= 0:
                                continue
                            img_path = os.path.join(folder_path, img_file)
                            if os.path.isfile(img_path):
                                self.images.append(img_path)
                                # 将文件夹名称作为标签
                                self.labels.append(1)

                    if label != '0':
                        i = 0
                        # img_files = os.listdir(folder_path)
                        # num_imgs_to_read = int(len(img_files) / num_file_to_read)
                        for img_file in os.listdir(folder_path):
                            # 只读取每个folder_path中的前num_imgs_to_read张图片
                            i = i + 1
                            if i > 200:
                                break
                            img_path = os.path.join(folder_path, img_file)
                            if os.path.isfile(img_path):
                                self.images.append(img_path)
                                # 将文件夹名称作为标签
                                self.labels.append(0)



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 根据索引获取图像路径
        img_path = self.images[idx]
        # 根据索引获取标签
        label = self.labels[idx]
        # 读取图像
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            # image = image.half()  # 转换为float16

        # 返回图像和对应的标签
        return image, torch.tensor(label)