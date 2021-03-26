"""
using spatial transformer network to calibrate cameras
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataloader import train_loader
import torchvision
from torchvision import transforms
from PIL import Image
from typing import Tuple
import cv2
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    """Function that returns Gaussian filter coefficients.

    Args:
        ksize (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(ksize,)`

    Examples::

        >>> tgm.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> tgm.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d


def get_gaussian_kernel2d(ksize: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    """Function that returns Gaussian filter matrix coefficients.

    Args:
        ksize (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(ksize_x, ksize_y)`

    Examples::

        >>> tgm.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> tgm.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class GaussianBlur(nn.Module):
    """Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = tgm.image.GaussianBlur((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float]) -> None:
        super(GaussianBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self._padding: Tuple[int, int] = self.compute_zero_padding(kernel_size)
        self.kernel: torch.Tensor = self.create_gaussian_kernel(
            kernel_size, sigma)

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""
        kernel: torch.Tensor = get_gaussian_kernel2d(kernel_size, sigma)
        return kernel

    @staticmethod
    def compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # convolve tensor with gaussian kernel
        return F.conv2d(x, kernel, padding=self._padding, stride=1, groups=c)


######################
# functional interface
######################


def gaussian_blur(src: torch.Tensor,
                  kernel_size: Tuple[int,
                                     int],
                  sigma: Tuple[float,
                               float]) -> torch.Tensor:
    """Function that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        src (Tensor): the input tensor.
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = tgm.image.gaussian_blur(input, (3, 3), (1.5, 1.5))
    """
    return GaussianBlur(kernel_size, sigma)(src)


class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self.gaussion = GaussianBlur((51, 51), (3.0, 3.0))
        self.params = nn.ParameterDict({'eps': nn.Parameter(torch.tensor([0.01, 0.01, 0.01, -0.1, 0.01, 0.01], dtype=torch.float).view(6, 1)),
        'kt': nn.Parameter(torch.tensor([720, 720, 610, 170], dtype=torch.float))})

    def forward(self, x, y, z):
        # x for source, y for target, z for depth nchw
        projected_img, valid_points=self.transform(x, y, z)
        diff=(y - projected_img) * valid_points.unsqueeze(1).float()
        loss=self.gaussion(diff)
        return nn.MSELoss(reduction='mean')(torch.zeros_like(loss, dtype=torch.float), loss)
    
    def toKmatrix(self):
        fx = self.params['kt'][0]
        fy = self.params['kt'][1]
        cx = self.params['kt'][2]
        cy = self.params['kt'][3]
        zero = torch.zeros_like(fx)
        one = torch.ones_like(fx)
        K = torch.stack([fx, zero, cx, zero, fy, cy, zero, zero, one]).to(device)
        return K.view(3, 3)
        
    def transform(self, x, y, z):
        """
        Inverse warp a source image(Thermal) to the target image(RGB) plane.
        Args:
            x: the source image (where to sample pixels) -- [N, 1, H, W]
            depth: depth map of the target image -- [N, H, W]
        Returns:
            projected_img: Source image warped to the target image plane
            valid_points: Boolean array indicating point validity
        """
        k1=torch.tensor([7.215377e+02, 0.000000e+00, 6.095593e+02, 00.000000e+00, 7.215377e+02,
                           1.728540e+02, 0, 0, 1], dtype=torch.float).reshape((3, 3)).unsqueeze(0).to(device)
        k2 = self.toKmatrix()
        E=self.se3toSE3(self.params['eps']).unsqueeze(0)
        cam_coords=self.pixel2cam(z, k1.inverse())  # [N,3,H,W]
        # Get projection matrix for tgt camera frame to source pixel frame
        proj_cam_to_src_pixel=k2 @ E  # [3, 4]
        rot, tr=proj_cam_to_src_pixel[...,
                                        :3], proj_cam_to_src_pixel[..., -1:]
        src_pixel_coords=self.cam2pixel(cam_coords, rot, tr)  # [B,H,W,2]
        projected_img=F.grid_sample(x, src_pixel_coords, align_corners=True)
        valid_points=src_pixel_coords.abs().max(dim=-1)[0] <= 1
        return projected_img, valid_points

    def se3toSE3(self, se3):
        # 6x1 to 3x4
        rho=se3[:3]
        psi=se3[3:]
        theta=torch.norm(psi)
        a=psi / theta
        zero=torch.zeros_like(a[1])
        ones=torch.eye(3).to(device)
        across=torch.stack(
            (zero, -a[2], a[1], a[2], zero, a[0], -a[1], a[0], zero)).view(3, 3)
        R=torch.cos(theta) * ones + (1-torch.cos(theta)) * \
            a@a.transpose(1, 0) + torch.sin(theta)*across
        jaccab=torch.sin(theta) / theta * ones + (1-torch.sin(theta) /
                                                    theta)*a@a.transpose(1, 0) + (1-torch.cos(theta)) / theta * across
        SE=torch.cat((R, jaccab@rho), dim=1)
        return SE

    def pixel2cam(self, depth, intrinsics_inv):
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [N, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [N, 3, 3]
        Returns:
            array of (x,y,z) cam coordinates -- [N, 3, H, W]
        """
        b, _, h, w=depth.size()
        i_range=torch.arange(0, h).view(1, h, 1).expand(
            1, h, w).type_as(depth).to(device)  # [1, H, W]
        j_range=torch.arange(0, w).view(1, 1, w).expand(
            1, h, w).type_as(depth).to(device)  # [1, H, W]
        ones=torch.ones(1, h, w).type_as(depth).to(device)
        pixel_coords=torch.stack(
            (j_range, i_range, ones), dim=1)  # [1, 3, H, W]
        current_pixel_coords=pixel_coords[..., :h, :w].expand(
            b, 3, h, w).reshape(b, 3, -1)  # [N, 3, H*W]
        cam_coords=(intrinsics_inv @
                      current_pixel_coords).reshape(b, 3, h, w)
        return cam_coords * depth

    def cam2pixel(self, cam_coords, proj_c2p_rot, proj_c2p_tr):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        b, _, h, w=cam_coords.size()
        cam_coords_flat=cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
        if proj_c2p_rot is not None:
            pcoords=proj_c2p_rot @ cam_coords_flat
        else:
            pcoords=cam_coords_flat

        if proj_c2p_tr is not None:
            pcoords=pcoords + proj_c2p_tr  # [B, 3, H*W]
        X=pcoords[:, 0]
        Y=pcoords[:, 1]
        Z=pcoords[:, 2].clamp(min=1e-3)

        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        X_norm=2*(X / Z)/(w-1) - 1
        Y_norm=2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

        pixel_coords=torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        return pixel_coords.reshape(b, h, w, 2)
    def get_EK(self):
        return self.se3toSE3(self.params['eps']).data.cpu(), self.params['kt'].data.cpu()


model=Transform().to(device)
optimizer=optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target, depth) in enumerate(train_loader):
        data, target, depth=data.to(device), target.to(
            device), depth.to(device)
        optimizer.zero_grad()
        output=model(data, target, depth)
        loss=output
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp=inp.numpy().transpose((1, 2, 0))
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    inp=std * inp + mean
    inp=np.clip(inp, 0, 1)
    return inp

def test():
    img_path = '/home/yidan/Work/sceneflow/data_scene_flow/training/image_3/000005_10.png'
    tgt_path = '/home/yidan/Work/sceneflow/data_scene_flow/training/image_2/000005_10.png'
    depth_path = '/home/yidan/Work/sceneflow/data_scene_flow/training/disp_occ_0/000005_10.png'

    img_tensor = transforms.ToTensor()(transforms.Grayscale(
        num_output_channels=1)(Image.open(img_path)))
    tgt_tensor = transforms.ToTensor()(transforms.Grayscale(
        num_output_channels=1)(Image.open(tgt_path)))
    img_tensor = transforms.Normalize((0.485,), (0.229,))(img_tensor).unsqueeze(0)
    tgt_tensor = transforms.Normalize((0.485,), (0.229,))(tgt_tensor).unsqueeze(0)
    disp_img = cv2.imread(depth_path, -1).astype(np.float) / 256.0
    disp = np.where(disp_img > 0, disp_img, 5)
    depth_img = 384.3814 / disp
    depth_tensor = transforms.ToTensor()(depth_img).float().unsqueeze(0)
    p, v = model.transform(img_tensor.to(
        device), tgt_tensor.to(device), depth_tensor.to(device))
    pg = torchvision.utils.make_grid(p.squeeze(0) * v.float())
    pi = convert_image_np(pg.data.cpu())
    plt.imshow(pi)
    plt.show()

if __name__ == '__main__':
    for epoch in range(1, 5):
        train(epoch)
    model.eval()
    with torch.no_grad():
        theta=model.get_EK()
        print('theta: ', theta)
    test()