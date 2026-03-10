import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (2, 3))
    a_01 = torch.sum(mask * prediction, (2, 3))
    a_11 = torch.sum(mask, (2, 3))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (2, 3))
    b_1 = torch.sum(mask * target, (2, 3))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


class SILogLoss_l1(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss_l1, self).__init__()
        self.name = 'L1_loss'

    def forward(self, input, target, mask=None, interpolate=True):
        # print('1: ',input)
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bicubic', align_corners=True)
        # scale, shift = compute_scale_and_shift(input.clone(), target.clone(), mask.clone())

        if mask is not None:
            input = input[mask]
            target = target[mask]

        # input = 1.0 / (input * 0.000040402)
        # target = ((1.0/target))/0.000040402
        # input = scale.view(-1, 1, 1, 1)*input + shift.view(-1, 1, 1, 1)

        # print('2: ',input)
        # print('22: ',target)

        return torch.mean(torch.abs(input - target))


class ScaleInvariantLoss(nn.Module):
    def __init__(self, penalty_weight=1000.0, alpha = 1):
        """
        初始化 ScaleInvariantLoss。

        参数:
            penalty_weight (float): 当 pred 包含负值时，负值惩罚项的权重。默认值为 1000.0。
        """
        super(ScaleInvariantLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.alpha = alpha

    def forward(self, pred, target, mask=None, interpolate=True):
        """
        前向传播计算损失。

        参数:
            pred (torch.Tensor): 预测的深度图，形状为 [B, 1, H, W]。
            target (torch.Tensor): 目标深度图，形状为 [B, 1, H, W]。
            mask (torch.Tensor, 可选): 掩码，形状为 [B, 1, H, W]，有效像素为1，无效为0。
            interpolate (bool): 是否对 pred 进行插值以匹配 target 的空间尺寸。

        返回:
            loss (torch.Tensor): 计算得到的损失值。
        """
        # 检查 pred 和 target 是否包含非有限值
        assert torch.isfinite(pred).all(), "Prediction contains non-finite values."
        assert torch.isfinite(target).all(), "Target contains non-finite values."
        # assert torch.sum(mask) > 5, "it is nearly all false in mask"

        # 可选的插值操作
        if interpolate:
            pred = nn.functional.interpolate(pred, target.shape[-2:], mode='bicubic', align_corners=True)

        # 应用掩码（如果提供）
        if mask is not None:
            pred = pred[mask == 1]
            target = target[mask == 1]

        # 检查 pred 是否包含任何负值
        has_negative = (pred < 0).any()

        if has_negative:
            # 提取所有负值
            negative_preds = pred[pred < 0]

            # 计算负值的惩罚项，这里使用负值的平方和
            penalty = torch.sum(negative_preds ** 2)

            # 将惩罚项乘以权重
            loss = self.penalty_weight * penalty

            return loss
        else:
            # 如果没有负值，则计算标准的尺度不变损失

            # 避免对数运算中的零值，添加一个小的偏移量
            eps = 1e-6
            diff_log = torch.log(pred + eps) - torch.log(target + eps)

            # diff_log 的形状为 [M]，M 为有效像素数
            n = diff_log.numel()

            if n == 0:
                # 如果没有有效像素，则返回零损失
                return torch.tensor(0.0, device=pred.device, requires_grad=True)

            diff_log_sum = torch.sum(diff_log)

            # 计算尺度不变损失
            loss = (torch.sum(diff_log ** 2) / n) - (self.alpha*(diff_log_sum ** 2) / (n ** 2))

            return loss

class SMS_Loss(nn.Module):
    def __init__(self, interpolate=True):
        super(SMS_Loss, self).__init__()
        self.name = 'SMS_Loss'
        self.interpolate = interpolate

    def gradient_x(self, img):
        # img: [B, 1, H, W]
        return img[:, :, :, 1:] - img[:, :, :, :-1]

    def gradient_y(self, img):
        return img[:, :, 1:, :] - img[:, :, :-1, :]

    def forward(self, input, target, mask=None, interpolate=True):
        # 可选的上采样，将input与target对齐尺寸
        if interpolate:
            input = nn.functional.interpolate(input, size=target.shape[-2:], mode='bicubic', align_corners=True)

        # diff: [B, 1, H, W]
        diff = input - target

        # 计算x、y方向梯度
        dx = self.gradient_x(diff)  # [B, 1, H, W-1]
        dy = self.gradient_y(diff)  # [B, 1, H-1, W]

        if mask is not None:
            # 如果有mask，则只对有效像素之间的梯度做统计
            # mask: [B, 1, H, W], 有效像素为1，无效为0
            # 对x方向梯度对应的mask
            mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]  # 相邻两像素都需有效
            mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

            valid_x = mask_x.sum()
            valid_y = mask_y.sum()

            loss_x = (torch.abs(dx) * mask_x).sum() / (valid_x + 1e-8)
            loss_y = (torch.abs(dy) * mask_y).sum() / (valid_y + 1e-8)
        else:
            # 没有mask，直接对所有像素梯度计算损失
            loss_x = torch.mean(torch.abs(dx))
            loss_y = torch.mean(torch.abs(dy))

        return loss_x + loss_y


class DepthSSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, K=(0.01, 0.03), size_average=True):
        """
        Args:
            window_size (int): Size of the Gaussian window. Default: 11
            sigma (float): Standard deviation of the Gaussian window. Default: 1.5
            K (tuple): Constants for SSIM formula. Default: (0.01, 0.03)
            size_average (bool): If True, average the SSIM over all pixels. Default: True
        """
        super(DepthSSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.K = K
        self.size_average = size_average
        self.padding = window_size // 2
        self.channel = 1  # Assuming single channel (depth maps)
        self.register_buffer('window', self.create_window(window_size, sigma))

    def create_window(self, size, sigma):
        # Create a 1D Gaussian kernel
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        # Create 2D Gaussian kernel
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, size, size]

        return window_2d

    def forward(self, input, target, mask=None, interpolate=True):
        """
        Args:
            input (torch.Tensor): Predicted depth map [B, 1, H, W]
            target (torch.Tensor): Ground truth depth map [B, 1, H, W]
            mask (torch.Tensor, optional): Mask indicating valid regions [B, 1, H, W]. Default: None
        Returns:
            torch.Tensor: SSIM loss
        """
        if interpolate:
            input = nn.functional.interpolate(input, size=target.shape[-2:], mode='bicubic', align_corners=True)

        if input.size() != target.size():
            raise ValueError("Input and target must have the same dimensions.")
        if mask is not None and mask.size() != input.size():
            raise ValueError("Mask must have the same dimensions as input and target.")

        B, C, H, W = input.size()
        window = self.window.type_as(input)

        # Apply mask if provided
        if mask is not None:
            # 将mask转换为浮点类型
            mask = mask.float()
            input = input * mask
            target = target * mask

        # Compute means
        mu_input = nn.functional.conv2d(input, window, padding=self.padding, groups=1)
        mu_target = nn.functional.conv2d(target, window, padding=self.padding, groups=1)

        if mask is not None:
            # Compute mask-weighted means
            mu_input = mu_input / (nn.functional.conv2d(mask, window, padding=self.padding, groups=1) + 1e-8)
            mu_target = mu_target / (nn.functional.conv2d(mask, window, padding=self.padding, groups=1) + 1e-8)

        mu_input_sq = mu_input.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_input_target = mu_input * mu_target

        # Compute variances and covariance
        sigma_input_sq = nn.functional.conv2d(input * input, window, padding=self.padding, groups=1) / (
                    nn.functional.conv2d(mask, window, padding=self.padding, groups=1) + 1e-8) - mu_input_sq
        sigma_target_sq = nn.functional.conv2d(target * target, window, padding=self.padding, groups=1) / (
                    nn.functional.conv2d(mask, window, padding=self.padding, groups=1) + 1e-8) - mu_target_sq
        sigma_input_target = nn.functional.conv2d(input * target, window, padding=self.padding, groups=1) / (
                    nn.functional.conv2d(mask, window, padding=self.padding, groups=1) + 1e-8) - mu_input_target

        # SSIM formula
        # 结构比较 s(x,y) = (sigma_xy + C3) / (sigma_x * sigma_y + C3)
        L = torch.max(input.max(), target.max())  # L 仍然用于计算 C3
        # C1 不再需要
        C2 = (self.K[1] * L) ** 2  # K[1] 是 K_2
        C3 = C2 / 2.0
        # 计算标准差 sigma_x 和 sigma_y
        # 添加 F.relu 以确保根号内不为负（由于浮点误差可能出现极小的负值）
        # 同时添加一个小的 epsilon 以增加数值稳定性
        sigma_input = torch.sqrt(F.relu(sigma_input_sq) + 1e-12)  # sigma_x
        sigma_target = torch.sqrt(F.relu(sigma_target_sq) + 1e-12)  # sigma_y

        # 结构比较的分子和分母
        numerator_s = sigma_input_target + C3
        denominator_s = sigma_input * sigma_target + C3

        structure_map = numerator_s / (denominator_s + 1e-8)  # 加 1e-8 防止除以零
        # ssim_map 现在只包含结构信息
        ssim_map = structure_map
        # --- 修改结束 ---

        # C1 = (self.K[0] * torch.max(input, target).max()) ** 2
        # C2 = (self.K[1] * torch.max(input, target).max()) ** 2
        #
        # numerator = (2 * mu_input_target + C1) * (2 * sigma_input_target + C2)
        # denominator = (mu_input_sq + mu_target_sq + C1) * (sigma_input_sq + sigma_target_sq + C2)
        #
        # ssim_map = numerator / (denominator + 1e-8)

        if self.size_average:
            # Average SSIM over all pixels and batch
            loss = 1 - ssim_map.mean()
        else:
            # Return SSIM map
            loss = 1 - ssim_map

        return loss


class MaskedEdgeLoss(nn.Module):
    """
    计算预测图像和目标图像在梯度（边缘）上的L1或L2损失，
    同时根据提供的掩码忽略无效区域。
    """

    def __init__(self, loss_type: str = 'l1'):
        """
        Args:
            loss_type (str): 应用于梯度差异的损失类型。
                             可选值为 'l1' (默认) 或 'l2'。
        """
        super(MaskedEdgeLoss, self).__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type not in ['l1', 'l2']:
            raise ValueError(f"不支持的损失类型: {loss_type}。请选择 'l1' 或 'l2'。")

    def _get_gradients(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算图像的水平和垂直梯度。
        Args:
            image (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)。
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - grad_x: 水平梯度，形状为 (B, C, H, W-1)。
                - grad_y: 垂直梯度，形状为 (B, C, H-1, W)。
        """
        # 计算x方向梯度 (水平)
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        # 计算y方向梯度 (垂直)
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        return grad_x, grad_y

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        计算带掩码的边缘损失。
        Args:
            prediction (torch.Tensor): 预测的深度图或图像，形状为 (B, C, H, W)。
            target (torch.Tensor): 真实的深度图或图像，形状为 (B, C, H, W)。
            mask (torch.Tensor): 二进制掩码，指示有效像素 (1 表示有效, 0 表示无效)，
                                 形状应为 (B, C, H, W) 或 (B, 1, H, W) 以支持广播。
        Returns:
            torch.Tensor: 表示计算得到的带掩码边缘损失的标量张量。
        """
        prediction = nn.functional.interpolate(prediction, size=target.shape[-2:], mode='bicubic', align_corners=True)
        if not (prediction.shape == target.shape):
            raise ValueError(f"预测张量形状 {prediction.shape} 和目标张量形状 {target.shape} 必须一致。")
        if not (prediction.dim() == 4):
            raise ValueError("输入张量 (prediction, target, mask) 必须是4D张量 (B, C, H, W)。")

        # 校验掩码形状是否与预测/目标张量兼容
        if not (mask.shape[0] == prediction.shape[0] and
                mask.shape[2:] == prediction.shape[2:] and
                (mask.shape[1] == prediction.shape[1] or
                 mask.shape[1] == 1)):
            raise ValueError(f"掩码形状 {mask.shape} 与预测张量形状 {prediction.shape} 不兼容。")

        # 如果掩码的通道数为1且预测张量的通道数大于1，则扩展掩码以匹配预测张量
        if mask.shape[1] == 1 and prediction.shape[1] > 1:
            mask = mask.expand_as(prediction)

        # 确保掩码是浮点类型，以便进行乘法和求和操作
        mask = mask.float()

        # 计算预测值和目标值的梯度
        pred_grad_x, pred_grad_y = self._get_gradients(prediction)
        target_grad_x, target_grad_y = self._get_gradients(target)

        # 为梯度创建更鲁棒的掩码：只有当构成梯度的两个像素都有效时，该梯度才被认为是有效的。
        # mask_x 的形状为 (B, C, H, W-1)
        mask_x = mask[:, :, :, :-1] * mask[:, :, :, 1:]
        # mask_y 的形状为 (B, C, H-1, W)
        mask_y = mask[:, :, :-1, :] * mask[:, :, 1:, :]

        # 计算梯度差异的损失值
        if self.loss_type == 'l1':
            loss_val_x = torch.abs(pred_grad_x - target_grad_x)
            loss_val_y = torch.abs(pred_grad_y - target_grad_y)
        else:  # self.loss_type == 'l2'
            loss_val_x = torch.pow(pred_grad_x - target_grad_x, 2)
            loss_val_y = torch.pow(pred_grad_y - target_grad_y, 2)

        # 将掩码应用于计算得到的损失值
        masked_loss_x = loss_val_x * mask_x
        masked_loss_y = loss_val_y * mask_y

        # 计算x和y方向梯度损失的总和
        sum_masked_loss_components = masked_loss_x.sum() + masked_loss_y.sum()

        # 计算梯度掩码中有效像素的总数
        # 添加一个小的epsilon以防止在掩码全为零时除以零
        epsilon = 1e-8
        num_valid_pixels_in_gradient_mask = mask_x.sum() + mask_y.sum() + epsilon

        # 通过有效像素总数对损失进行归一化
        total_loss = sum_masked_loss_components / num_valid_pixels_in_gradient_mask

        return total_loss


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        # print('1: ',input)
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        # scale, shift = compute_scale_and_shift(input.clone(), target.clone(), mask.clone())

        if mask is not None:
            input = input[mask]
            target = target[mask]
        # input = scale.view(-1, 1, 1, 1)*input + shift.view(-1, 1, 1, 1)
        # print('2: ',input)
        # print('22: ',target)
        epsilon = 1e-6
        g = torch.log(input + epsilon) - torch.log(target + epsilon)
        # print('3: ',g)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss
