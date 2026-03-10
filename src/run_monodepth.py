"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse
import dataset_util as dsutil
import util.io
import model_io
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from util.misc import visualize_attention


def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
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
            scale=0.000040402,
            shift=0.01,
            invert=True,
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

    transform = Compose(
        [
            Resize(
                768,
                384,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=384,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_LANCZOS4,
            ),
            # normalization,
            PrepareForNet(),
        ]
    )
    pretrained_path = model_path
    # pretrained_path = (
    #     "/mnt/dev/projects/edric/DPT_Gated_trianer/"
    #     "checkpoints/13-Jan_10-56-nodebs1-tep100-lr0.0001-wd0.01-d4c60633-60f7-4a5f-8a1e-dd98577ceb64/DPT_gated_RGB_epoch8.pt")
    rst_fn = output_path
    if not os.path.exists(rst_fn):
        os.makedirs(rst_fn)
    output_path = os.path.join(rst_fn, pretrained_path.split("/")[-2]+pretrained_path.split("/")[-1][-11:-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model, _, _ = model_io.load_checkpoint(pretrained_path, model)
    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_path, 'gated0_10bit', "*"))
    num_images = len(img_names)

    # create output folder
    # os.makedirs(output_path, exist_ok=True)

    g2d_crop_height = 420
    g2d_crop_width = 980
    height_offset = 0
    g2d_crop_height_offset = int((720 - g2d_crop_height) / 2) + height_offset
    g2d_crop_width_offset = int((1280 - g2d_crop_width) / 2)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input
        # img = util.io.read_image(img_name)
        img = dsutil.read_gated_image(input_path, img_name[-9:-4])
        img = img[g2d_crop_height_offset:g2d_crop_height_offset + g2d_crop_height,
              g2d_crop_width_offset:g2d_crop_width_offset + g2d_crop_width, :]
        img_input = transform({"image": img})["image"]
        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            # prediction = 1.0/(prediction * 0.000040402)

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_depth(filename, prediction, bits=2, absolute_depth=args.absolute_depth)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')

    parser.add_argument(
        # "-i", "--input_path", default="figs/input_night_test_samples", help="folder with input images"
        "-i", "--input_path", default="figs/input_night_final", help="folder with input images"
        # "-i", "--input_path", default="/mnt/dev/datasets/gated2depth/data/real", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="figs/output_night_test_samples",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights",
        default=
        '/mnt/dev/projects/gjm/DPT_Gated_trianer_new/'
        'checkpoints/04-Jun_13-07-lidaronly-vh-bs1/DPT_gated_RGB_epoch6.pt',
        help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )
