import os

import numpy as np
import torch
from matplotlib import pyplot as plt
# import cupyx.matplotlib.pyplot as plt
from torchvision import transforms
from dataloader import DepthDataLoader, _is_pil_image, _is_numpy_image
import model_io
from PIL import Image
from tqdm import tqdm
import matplotlib
import dataset_util as dsutil
matplotlib.use('agg')

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 150
MAX_DEPTH_KITTI = 150

N_BINS = 256
input_width = 1000
input_height = 600

gta_pass = ''
data_type = 'real'
min_distance = 3.
max_distance = 150.
g2d_crop_height = 420
g2d_crop_width = 1200
g2d_crop_height_offset = int((720 - g2d_crop_height) / 2)
g2d_crop_width_offset = int((1280 - g2d_crop_width) / 2)

def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError(
            'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)).copy())
        return img

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

data_path = '/mnt/dev/datasets/gated2depth/data/real/splits/'
# data_path = '/mnt/hdd/LYP/gated2gated/data/splits/stf'
daytime = 'night'
fpath = os.path.join(data_path, "real_test_{}.txt".format(daytime))
# fpath = os.path.join(data_path, "test_clear_{}.txt".format(daytime))


train_filenames = readlines(fpath)
pretrained_path = (
    "/mnt/dev/projects/edric/DPT_Gated_trianer/checkpoints/17-Dec_16-08-drr_sil10-tep50-lr5e-05-wd0.01-55bd1ca1-228d-4276-a800-707f301fef2d/DPT_gated_RGB_best.pt")

base_dir = '/mnt/d/datasets/gated2depth/data/real/'
rst_fn = 'test_outputs'

if not os.path.exists(rst_fn):
    os.makedirs(rst_fn)
image_dir = os.path.join(rst_fn, pretrained_path.split("/")[-1][0:-3])
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
result_path = os.path.join(image_dir, daytime)
if not os.path.exists(result_path):
    os.makedirs(result_path)

# load network
    model_type = "dpt_hybrid"
    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }
    # model_path = default_models[model_type]
    model_path = None
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"



model, _, _ = model_io.load_checkpoint(pretrained_path, model)
model.eval()


for img_id in tqdm(train_filenames):
    image = dsutil.read_gated_image(base_dir, img_id)
    example_rgb_batch = image[
                        g2d_crop_height_offset:g2d_crop_height_offset + g2d_crop_height,
                        g2d_crop_width_offset:g2d_crop_width_offset + g2d_crop_width]
    # example_rgb_batch = os.path.join(base_dir, img_id + '.png')
    # example_rgb_batch = Image.open(example_rgb_batch)
    # example_rgb_batch = example_rgb_batch.crop(((1280 - input_width) / 2,
    #                                             (720  - input_height) / 2,
    #                                             (1280 - input_width) / 2 + input_width,
    #                                             (720   - input_height) / 2 + input_height))
    # example_rgb_batch = np.asarray(example_rgb_batch, dtype=np.float32) / 255.0

    example_rgb_batch = to_tensor(example_rgb_batch)
    # example_rgb_batch = normalize(example_rgb_batch)
    example_rgb_batch = torch.unsqueeze(example_rgb_batch, 0)
    bin_edges, predicted_depth = model(example_rgb_batch.to(device='cuda'))
    predicted_depth = predicted_depth.permute((0, 2, 3, 1))[0, :, :, 0].cpu().detach().numpy()

    # print(predicted_depth.shape)
    plt.imshow(predicted_depth, cmap='jet_r')

    # plt.scatter(x_coords, y_coords, c=150 * (gt_patch.reshape(figShape)[lidar_mask.reshape(figShape)]),
    #             cmap='jet_r', s=9, marker='o', edgecolors='white', linewidths=0.5)
    # plt.colorbar()
    # plt.title("Predicted Depth Map")
    plt.axis('off')
    plt.savefig(os.path.join(result_path, img_id + '.png'), bbox_inches='tight', pad_inches=-0.0)
    plt.clf()
    # break

# KITTI
# model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
# pretrained_path = "./pretrained/AdaBins_kitti.pt"
# model, _, _ = model_io.load_checkpoint(pretrained_path, model)
#
# bin_edges, predicted_depth = model(example_rgb_batch)
print('done')
