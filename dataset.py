import torch
import imageio
import os
import numpy as np
from tqdm.auto import tqdm
from structured_uncertainty.transforms import rescale_to
from scipy.ndimage.morphology import distance_transform_edt

from torchvision.transforms import Resize

class CellCrops(torch.utils.data.Dataset):
    r"""
    Loads all cells into memory, they should be very small - cropped around
    RBCs. There is no ground truth, so we only load the raw images (used for
    autoenoder training).
    """
    
    def __init__(self, img_path, transforms=None, ext=None, load_to_gpu=0, set_size=128, debug=False):
        r"""
        Args:
            :img_path: str/path. Points to the folder containing the images.
        """
        self.transforms = transforms

        self.set_size = set_size
        self.img_size = (set_size, set_size)
        self.cell_names = os.listdir(img_path)
        if isinstance(debug, bool):
            self.debug = debug
            self._debug_len = 500
        else:
            self.debug = True
            self._debug_len = debug

        if debug:
            self.dset_len = self._debug_len
        else:
            self.dset_len = len(self.cell_names)

        if ext is not None:
            self.cell_names = [name for name in cell_names if ext in name]

        # Sorting is not necessary, but might aid debugging later.
        self.cell_names.sort()

        if debug:
            self.cell_names = self.cell_names[:self._debug_len]

        self.cell_images = self._load_cells_from_names(img_path, self.cell_names)
        if load_to_gpu is not None:
            self.cell_images = self.cell_images.to()
        self.img_shape = self._get_image_shape()

    def __len__(self):
        return self.dset_len

    def _get_image_shape(self):
        return self.cell_images[0].shape

    def _load_cells_from_names(self, img_path, cell_names):
        r"""
        Load the images as np arrays, then turns them into float32.
        """
        temp_img = imageio.imread("{}/{}".format(img_path, cell_names[0]))
        unsq = False
        if len(temp_img.shape) == 2:
            # Image is grayscale, so no channel is loaded.
            outp_shape = (1, *self.img_size)
            unsq = True
        elif len(temp_img.shape) == 3:
            # Expect a RGB image, channels first
            outp_shape = (3, *self.img_size)
        else:
            raise Exception("Loaded unsupported img size. Expected Gray or RGB (channels first)")

        if self.debug:
            temp_array = torch.zeros((self._debug_len, *outp_shape))
        else:
            temp_array = torch.zeros((self.__len__(), *outp_shape)) # (N,C,*)
        resize = Resize(self.img_size)

        if not unsq:
            for idx, img_name in enumerate(tqdm(self.cell_names, desc="Loading crops...")):
                temp_array[idx] = resize(
                        torch.from_numpy(
                            imageio.imread(
                                "{}/{}".format(img_path, img_name))).float().transpose(0,-1)
                        )
        if unsq:
            for idx, img_name in enumerate(tqdm(self.cell_names, desc="Loading crops...")):
                temp_array[idx] = resize(
                        torch.from_numpy(
                            imageio.imread(
                                "{}/{}".format(img_path, img_name))).float().transpose(0,-1).unsqueeze(0)
                        )

        return temp_array

    def __getitem__(self, idx):
        r"""
        Args:
            :idx: int

        Returns:
            [C,*], float32 image array.
        """
        if self.transforms is None:
            return self.cell_images[idx]
        else:
            return self.transforms(self.cell_images[idx])

class FullFrames(torch.utils.data.Dataset):
    r"""
    Loads the frames on demand. A bit slower, but there can be a large frame 
    dataset. Expecting images to be uint8, shape (H,W) when loaded by imageio.
    """
    
    def __init__(self, img_path, gt_path, raw_transforms=None, joint_transforms=None, ext=None, load_all=False, debug=False, apply_joint_first=True):
        r"""
        Args:
            :img_path: str/path. Points to the folder containing the images.
            :gt_path: str/path to ground truth masks. Should have the same names as in img_path.
        """
        self.raw_transforms = raw_transforms
        self.joint_transforms = joint_transforms
        self.apply_joint_first = apply_joint_first

        self.img_path = img_path
        self.gt_path = gt_path
        self.cell_names = os.listdir(img_path)
        self.cell_names.sort()

        if isinstance(debug, bool):
            self.debug = debug
            self._debug_len = 500
        else:
            self.debug = True
            self._debug_len = debug

        if debug:
            self.dset_len = self._debug_len
        else:
            self.dset_len = len(self.cell_names)

        # If debug, dset_len is much smaller than the real one (to speed up things).
        self.cell_names = self.cell_names[:self.dset_len]

        if ext is not None:
            self.cell_names = [name for name in cell_names if ext in name]

        self.load_all = load_all
        if load_all:
            self.raw_cells = []
            self.gt_cells = []
            for cell_name in tqdm(self.cell_names, desc="Loading all data to RAM.."):
                self.raw_cells.append(
                    torch.from_numpy(rescale_to(imageio.imread("{}/{}".format(self.img_path, cell_name)), to=(0., 255.)).astype("uint8")).float().unsqueeze(0)
                        )
                self.gt_cells.append(
                    torch.from_numpy(rescale_to(imageio.imread("{}/{}".format(self.gt_path, cell_name)), to=(0., 255.)).astype("uint8")).float().unsqueeze(0)
                        )

    def __len__(self):
        return self.dset_len

    def _get_image_shape(self):
        if self.load_all:
            assert len(self.raw_cell) > 0, "No images loaded.."
            return self.raw_cells[0].shape
        else:
            tmp_ = imageio.imread("{}/{}".format(self.img_path, self.cell_names[0]))
            return tmp_.shape

    def __getitem__(self, idx):
        r"""
        Args:
            :idx: int

        Returns:
            [C,*], float32 image array.
        """
        if self.load_all:
            img = self.raw_cells[idx]
            gt = self.gt_cells[idx]
        else:
            img = torch.from_numpy(rescale_to(imageio.imread("{}/{}".format(self.img_path, self.cell_names[idx])), to=(0., 255.)).astype("uint8")).float().unsqueeze(0)
            gt = torch.from_numpy(imageio.imread("{}/{}".format(self.gt_path, self.cell_names[idx]))).unsqueeze(0)

        if self.apply_joint_first:
            if self.joint_transforms is not None:
                for jtform in self.joint_transforms:
                    img, gt = jtform(img, gt)
            if self.raw_transforms is not None:
                img = self.raw_transforms(img)
        else:
            if self.raw_transforms is not None:
                img = self.raw_transforms(img)
            if self.joint_transforms is not None:
                for jtform in self.joint_transforms:
                    img, gt = jtform(img, gt)

        return img, gt

class MvtecClass(torch.utils.data.Dataset):
    r"""
    Loads a Mvtec dataset class. All images loaded at once, there aren't very
    many of them. Each channel is treated as a separate image.
    """
    
    def __init__(self, img_path, transforms=None, ext=".png", debug=False,
            resize_to=(128,128), recursive_except=None, load_with_rgb=False, fake_gt=False):
        r"""
        Args:
            :img_path: str/path. Points to the folder containing the images.
            :gt_path: str/path to ground truth masks. Should have the same names as in img_path.
            :load_with_rgb: bool, if True, will load images as [3,H,W] if they
                are in colour, or will expand the gray image into [3,H,W]. If 
                False, then grayscales will be loaded as grayscale [1,H,W], and 
                RGBs will have each channel loaded as a separate image.

        Use to get the test set:

        #dset = MvtecClass("/u/homes/biv20/datasets/mvtek/bottle/test", ext=".png", recursive_except=['good'])

        """
        self.fake_gt = fake_gt
        self.transforms = transforms
        self.img_path = img_path
        self.img_names = os.listdir(img_path)
        self.img_names.sort()
        self.load_with_rgb = load_with_rgb

        self.image_size = (1, resize_to[0], resize_to[1])
        self.resize = Resize(resize_to)

        if isinstance(debug, bool):
            self.debug = debug
            self._debug_len = 10
        else:
            self.debug = True
            self._debug_len = debug

        self.raw_imgs = []
        if recursive_except is None:

            for cell_name in tqdm(self.img_names, desc="Loading all data to RAM.."):
                img_rgb = imageio.imread("{}/{}".format(self.img_path, cell_name))
                self.process_imageio_input(img_rgb)

        else:
            self._load_recursive(img_path, skip=recursive_except, ext=ext)

        self.dset_len = len(self.raw_imgs)
 
    def process_imageio_input(self, img):
        r"""
        Adds the loaded image to the raw images list. It has to be turned into
        a torch Tensor, and the image size (whether it is gray or rgb, along
        with the load_with_rgb flag, determine how the image is loaded.
        """
        if len(img.shape) == 3 and not self.load_with_rgb:
            # assuming the last dim is the channels, add each channel as 
            # separate gray image.
            for ch in range(img.shape[-1]):
                self.raw_imgs.append(
                    self.resize(
                        torch.from_numpy(rescale_to(img[...,ch], to=(0., 255.)).astype("uint8")).float().unsqueeze(0)))
        elif len(img.shape) == 2 and self.load_with_rgb:
            # Fake rgb by expanding gray dim to 3 dims.
            self.raw_imgs.append(
                self.resize(
                torch.from_numpy(rescale_to(img, to=(0., 255.)).astype("uint8")).float().unsqueeze(0)).expand(3,-1,-1))
        elif len(img.shape) == 2 and not self.load_with_rgb:
            # Grayscale
            self.raw_imgs.append(
                self.resize(
                torch.from_numpy(rescale_to(img, to=(0., 255.)).astype("uint8")).float().unsqueeze(0)))
 
        elif len(img.shape) == 3:
            # Load rgb, or any number of other channels.
            self.raw_imgs.append(
                self.resize(
                torch.from_numpy(rescale_to(img, to=(0., 255.)).astype("uint8")).float().transpose(1,2).transpose(0,1)))

        else:
            raise Exception("Unrecognised image dim {} with rgb {}".format(
                img.shape, self.load_with_rgb))

    def __len__(self):
        return self.dset_len

    def _get_num_channels(self):
        return 1

    def _get_image_shape(self):
        return self.image_size

    def _load_recursive(self, root_dir, ext, skip=None):
        r"""
        For when the anomalous images live in different folders with the same
        root, and we want to load them, but skip the folder containing the
        normal images, given in the 'skip' list.
        """
        images = []

        for root, dirs, files in os.walk(root_dir):

            # Check that the root is not the folder we want to skip.
            if skip is not None and root.split("/")[-1] in skip:
                continue

            # Load the files with the correct extension.
            for f_ in files:
                if ext in f_:
                    img = imageio.imread("{}/{}".format(root,f_))
                    self.process_imageio_input(img)

            print("Loaded from {}".format(root))

    def __getitem__(self, idx):
        r"""
        Args:
            :idx: int

        Returns:
            [C,*], float32 image array.
        """
        img = self.raw_imgs[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        if self.fake_gt:
            return img, -1
        else:
            return img
