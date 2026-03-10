# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import dataset_util as dsutil
import FMW as fmw
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import cv2
from torchvision.transforms import Compose


# random.seed('123')


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DepthDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(300)
        min_distance = self.args.min_depth
        max_distance = self.args.max_depth
        # (1024 - y) / 2 > 130, y < 764
        # g2d_crop_height = random.randin9120323298394961t(400, 440)
        g2d_crop_height = 420
        # (1920 - x) / 2 > 100, x < 1724
        # g2d_crop_width = random.randint(960, 1000)
        g2d_crop_width = 980
        if self.mode == 'train':
            height_offset = random.randint(-50, 50)
            width_offset = random.randint(-60, 60)
            # height_offset = 0
            # width_offset = 0
        else:
            height_offset = 0
            width_offset = 0
        g2d_crop_height_offset = int((720 - g2d_crop_height) / 2) + height_offset
        g2d_crop_width_offset = int((1280 - g2d_crop_width) / 2) + width_offset

        if self.mode == 'train':
            img_id = sample_path[:-1]

            # image_path = os.path.join(self.args.data_path, 'rgb_left_8bit', img_id + '.png')
            # image = Image.open(image_path)
            image = dsutil.read_gated_image(self.args.data_path, img_id)
            depth_gt, _ = dsutil.read_gt_image(self.args.data_path, img_id, min_distance, max_distance)
            # depth_gt, _ = dsutil.read_gt_image_drr_denoised(self.args.data_path, img_id, min_distance, max_distance)
            # depth_gt = Image.open(os.path.join(self.args.data_path, "drr", img_id+".png"))


            # depth_gt = np.clip(
            #     np.asarray(
            #         Image.open(
            #             os.path.join(self.args.gt_path, img_id + '.png'))),
            #     min_distance,
            #     max_distance)

            rgb_depth = np.load(os.path.join(
                self.args.data_path,
                'depthAT-L_rgbl_projTo_gated',
                img_id + '.npz'))['arr0']
            # rgb_depth = np.load(os.path.join(
            #     self.args.data_path,
                # 'dpt_outputs_from_rgbl_cropped_finetune01_projTo_gated',
            #     img_id + '.npz'))['arr0']
            # print(image.shape)
            # print(depth_gt.shape)

            image = np.asarray(image, dtype=np.float32)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            rgb_depth = np.asarray(rgb_depth, dtype=np.float32)

            image = image[
                    g2d_crop_height_offset:g2d_crop_height_offset + g2d_crop_height,
                    g2d_crop_width_offset:g2d_crop_width_offset + g2d_crop_width, :]
            depth_gt = depth_gt[
                       g2d_crop_height_offset:g2d_crop_height_offset + g2d_crop_height,
                       g2d_crop_width_offset:g2d_crop_width_offset + g2d_crop_width]
            rgb_depth = rgb_depth[
                        g2d_crop_height_offset:g2d_crop_height_offset + g2d_crop_height,
                        g2d_crop_width_offset:g2d_crop_width_offset + g2d_crop_width]

            depth_gt = np.expand_dims(depth_gt, axis=2)
            rgb_depth = np.expand_dims(rgb_depth, axis=2)

            # image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'img_id': img_id, 'rgb_depth': rgb_depth, 'id': img_id}

        else:
            img_id = sample_path[:-1]
            image_path = os.path.join(self.args.data_path, 'rgb_left_8bit', img_id + '.png')
            # image = Image.open(image_path)
            image = dsutil.read_gated_image(self.args.data_path, img_id)
            image = np.asarray(image, dtype=np.float32)
            image = image[
                    g2d_crop_height_offset:g2d_crop_height_offset + g2d_crop_height,
                    g2d_crop_width_offset:g2d_crop_width_offset + g2d_crop_width, :]
            # print(image.shape)

            if self.mode == 'online_eval':
                has_valid_depth = False
                try:
                    depth_gt, _ = dsutil.read_gt_image(self.args.data_path, img_id, min_distance, max_distance)
                    # depth_gt, _ = dsutil.read_gt_image_drr_denoised(self.args.data_path, img_id, min_distance, max_distance)
                    depth_gt = depth_gt[
                               g2d_crop_height_offset:g2d_crop_height_offset + g2d_crop_height,
                               g2d_crop_width_offset:g2d_crop_width_offset + g2d_crop_width]
                    has_valid_depth = True
                except IOError:
                    depth_gt = False

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path[:-1] + '.png', 'depth_path': sample_path[:-1] + '.npz'}
            else:
                sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def dp_crop(self, dp_mask):
        if self.args.dataset == 'g2d':
            position, _ = fmw.max_true_area(dp_mask, width=self.args.input_width, height=self.args.input_height)
            return position

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # 禁用，因为这里的图像并非RGB
        # Random gamma, brightness, color augmentation
        # do_augment = random.random()
        # if do_augment > 0.5:
        #     image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resizer = Compose(
            [
                Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=384,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_LANCZOS4,
                )
            ]
        )

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.resizer({"image": image})["image"]
        image = self.to_tensor(image)
        # image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']

        if self.mode == 'train':
            rgb_depth = sample['rgb_depth']
            depth = self.to_tensor(depth)
            rgb_depth = self.to_tensor(rgb_depth)
            return {'image': image, 'depth': depth, 'focal': focal, 'rgb_depth': rgb_depth, 'id':sample['id']}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)).copy())
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
