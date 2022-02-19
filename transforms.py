import torch
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

def rescale_to(img, to=(0., 255.), eps_=1e-6):
    r"""
    :img: [B,C,*]
    """
    outp_min = to[0]
    outp_max = to[1]

    outp_ = img - img.min()
    outp_ = outp_ / (outp_.max() + eps_)
    outp_ = outp_ * (outp_max - outp_min)
    outp_ = outp_ + outp_min

    return outp_

def random_channel_select(img):
    r"""
    Input is an image with multiple channels, this function selects one of them
    at random and passes it on.

    Inputs:
        :img: (C,H,W), torch.Tensor
    """
    n_ch = img.shape[0]
    ch = torch.randint(n_ch, (1,)).item()

    return img[ch].unsqueeze(0)

def channel_select(img, ch=0):
    r"""
    Select channel of image, a naive way to turn an image grayscale.

    Inputs:
        :img: (C,H,W), torch.Tensor
    """
    return img[ch].unsqueeze(0)

def add_gaussian_noise(img, mean=0.01, std=0.01):
    r"""
    Add some gaussian noise to the image.
    """
    sample = torch.randn_like(img)*std + mean
    return img + sample

def random_gaussian_noise(img, mean=[-0.35,0.35], std=[0.01,0.3], p=0.5):
    r"""
    Sample parameters uniformly from the range and then sample the noise.

    Args:
        :img: torch.Tensor [B,(1,)H,W]
        :mean: float range
        :std: float range
        :p: float, probability (uniform) of applying this transform.
    """
    if torch.rand(1).item() < p:
        return img

    mean_range_width = mean[1] - mean[0]
    mean_ = (torch.rand(1).item() * mean_range_width) + mean[0]

    std_range_width = std[1] - std[0]
    std_ = (torch.rand(1).item() * std_range_width) + std[0]

    img = add_gaussian_noise(img, mean=mean_, std=std_)

    return img

def joint_random_crop(img, gt, side=128):
    r"""
    Select a random crop from the raw and gt jointly.

    Inputs:
        :img: (C,H,W)
        :gt: (C,H,W)
    """
    H,W = img.shape[-2:] # Last two dims should be spatial

    assert H > side and W > side

    h = torch.randint(H - side, (1,)).item()
    w = torch.randint(W - side, (1,)).item()

    img_crop = img[..., h : (h + side), w : (w + side)]
    gt_crop = gt[..., h : (h + side), w : (w + side)]

    return img_crop, gt_crop

def gt_sdf_normalization(raw, gt, side=128):
    return raw, gt/(side * 1.414)

def gt_sdf_sigmoid(raw, gt, scale=1):
    return raw, torch.nn.functional.sigmoid(gt / scale)

def gt_sdf_gaussian(raw,gt,scale=1):
    return raw, (-((gt/scale).square())).exp()

def sdt(img):

    mask_in = img != 0
    mask_out = img == 0

    outp = distance_transform_edt(mask_in) - distance_transform_edt(mask_out)

    return torch.from_numpy(outp)

def to_fg_bg_mask(img, gt, background_id=0):
    r"""
    Takes an instance segmentation mask and returns a semantic segmentation
    where :background_id: is the background and 1. is the foreground.
    """
    return img, (gt != background_id).float()

def _in_boundary(cent_h, cent_w, img_H, img_W, box_half_side):
    r"""
    Check if the bounding box will be within bounds of the frame.
    """
    return (cent_h > box_half_side and cent_w > box_half_side and
            cent_h + box_half_side < img_H and cent_w + box_half_side < img_W)

def joint_random_cell_crop(x, gt, box_half_side=64, background_id=0):
    r"""
    Crop an area around a cell based on the gt mask.

    Args:
        :x: [C,H,W]
        :gt: [1,H,W]
    """
    C, img_H, img_W = x.shape

    uniques_ = torch.unique(gt)
    while True:
        # while loop to skip over any ids that are out of bounds or the background.
        rand_idx = torch.randint(0, len(uniques_), (1,)).item()
        cell_id = uniques_[rand_idx]

        if cell_id == background_id:
            #skip
            continue

        cent_h, cent_w = np.argwhere( gt[0] == cell_id ).float().mean(axis=1).int()

        if not _in_boundary(cent_h, cent_w, img_H, img_W, box_half_side):
            #skip
            continue

        upper_left_h = cent_h - box_half_side
        upper_left_w = cent_w - box_half_side

        box_raw = rescale_to(np.copy(x[:, upper_left_h : upper_left_h + 2*box_half_side,\
                upper_left_w : upper_left_w + 2*box_half_side])).astype('uint8')
        box_raw = torch.from_numpy(box_raw)

        box_mask = np.copy(gt[:, upper_left_h : upper_left_h + 2*box_half_side,\
                upper_left_w : upper_left_w + 2*box_half_side]).astype('uint8')
        box_mask = torch.from_numpy(box_mask)
        # Remove other cell fragments from mask
        box_mask[box_mask != cell_id] = background_id
        box_mask[box_mask == cell_id] = 1.

        return box_raw, box_mask

class SigmoidScaleShift(torch.nn.Module):
    def __init__(self, scale=1., shift=0., len_scale=1., offset=0.):
        r"""
        Shifted and scaled Sigmoid function in y

        Args:
            :scale: float, multiplier of the sigmoid output
            :shift: float, displacement of the sigmoid output
        """
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.len_scale = len_scale # of X
        self.offset = offset # in X

    def forward(self, x):
        factor_ = -(x - self.offset)/self.len_scale
        sigmoid_ = torch.sigmoid(factor_)
        outp_ = self.shift + sigmoid_ * self.scale 

        # TODO Torch stability improvement?
        #logvar = self._sigmoid_log_range * 2.0 * (torch.sigmoid(logvar) - 0.5)
        return outp_
