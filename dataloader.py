import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CalibDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.img_list = []
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                if name.split('.')[-1] == 'png' and name.split('_')[-1] == '10.png':
                    left_name = os.path.join(root, name)
                    self.img_list.append(left_name)
        self.len = len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        index = idx % self.len
        source_name = self.img_list[index]
        target_name = source_name.replace('image_3', 'image_2')
        depth_name = source_name.replace('image_3', 'disp_occ_0')
        disp_img = cv2.imread(depth_name, -1)
        disp_img = disp_img.astype(np.float) / 256.0
        disp = np.where(disp_img > 0, disp_img, 5)
        depth_img = 384.3814 / disp
        depth_tensor = transforms.ToTensor()(depth_img).float()
        s_img = Image.open(source_name)
        t_img = Image.open(target_name)
        s_img = transforms.Grayscale(num_output_channels=1)(s_img)
        # s_img = transforms.Resize((512, 640))(s_img)
        s_img = transforms.ToTensor()(s_img)
        t_img = transforms.Grayscale(num_output_channels=1)(t_img)
        # t_img = transforms.Resize((512, 640))(t_img)
        t_img = transforms.ToTensor()(t_img)
        s_tensor = self.image_grad(s_img)
        t_tensor = self.image_grad(t_img)
        # s_tensor = self.transform(gs)
        # t_tensor = self.transform(gt)
        return (s_tensor, t_tensor, depth_tensor)

    def image_grad(self, img):
        # img = img.squeeze(0)
        # ten = torch.unbind(img)
        # x = ten[0].unsqueeze(0).unsqueeze(0)
        x = img.unsqueeze(0)  # CHW
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(
            a).float().unsqueeze(0).unsqueeze(0))
        G_x = conv1(torch.autograd.Variable(x)).data.view(
            1, x.shape[2], x.shape[3])
        b = np.array([[1, 2, 1], [0, 0, 0], [1, 0, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(
            b).float().unsqueeze(0).unsqueeze(0))
        G_y = conv2(torch.autograd.Variable(x)).data.view(
            1, x.shape[2], x.shape[3])
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        return G


# Training dataset
train_loader = torch.utils.data.DataLoader(
    CalibDataset(root_dir='/home/yidan/Work/sceneflow/data_scene_flow/training/image_3'), batch_size=8, shuffle=True, num_workers=4)

if __name__ == '__main__':
    import torchvision
    from matplotlib import pyplot as plt

    def convert_image_np(inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp
    # data = next(iter(train_loader))
    # imgs = torchvision.utils.make_grid(data[0])
    # imgt = torchvision.utils.make_grid(data[1])
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(convert_image_np(imgs))
    # axarr[0].set_title('source')
    # axarr[1].imshow(convert_image_np(imgt))
    # axarr[1].set_title('target')
    # plt.show()
    for batch_idx, (data, target, depth) in enumerate(train_loader):
        data, target, depth = data.to(
            device), target.to(device), depth.to(device)
