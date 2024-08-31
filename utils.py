from torchvision.utils import make_grid , save_image
import numpy as np
import torch
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
# 设置环境变量，允许程序继续执行，但可能会有风险
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class TopHalfCrop:
    def __call__(self, img):
        width, height = img.size
        return img.crop((0, 0, width, height // 2))
def show_and_save(file_name,img):
    # npimg = np.transpose(img.numpy(),(1,2,0))
    # f = "./%s.png" % file_name
    # fig = plt.figure(dpi=200)
    # fig.suptitle(file_name, fontsize=14, fontweight='bold')
    # plt.imshow(npimg)
    # plt.imsave(f,npimg)

    # 确保img是一个PyTorch张量
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a PyTorch Tensor")
    # 将张量从GPU移动到CPU，如果它在GPU上的话
    img = img.cpu()
    # 确保张量是浮点数，并且数值在[0, 1]范围内
    if img.dtype != torch.float32:
        img = img.float()
    img = torch.clamp(img, 0, 1)  # 确保数值在0到1之间
    # 将PyTorch张量转换为NumPy数组
    npimg = img.numpy()
    # 调整维度顺序以适应matplotlib的图像显示要求
    npimg = np.transpose(npimg, (1, 2, 0))
    # 保存图像的文件路径
    f = f"./{file_name}.png"
    # 创建图像
    fig = plt.figure(dpi=200)
    fig.suptitle(file_name, fontsize=14, fontweight='bold')
    # 显示图像
    plt.imshow(npimg)
    # 保存图像
    plt.imsave(f, npimg)
    # 关闭图形界面，以避免阻塞
    plt.close(fig)
def plot_loss(loss_list, filename):
    plt.figure(figsize=(10,5))
    plt.plot(loss_list,label="Loss")
    plt.legend()
    dot_index = filename.rfind('.')
    # 使用字符串的切片操作分离出前面的名称
    name = filename[:dot_index]
    plt.title(name)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.grid(True)  # 可以添加网格线
    plt.savefig(filename)  # 保存图像
    plt.show()
