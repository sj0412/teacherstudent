import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import cv2
from torchvision.transforms import Compose

import allArgs

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from matplotlib import pyplot as plt

import wandb
from tqdm import tqdm

import model_io
# import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss, SILogLoss_l1, SMS_Loss, ScaleInvariantLoss, DepthSSIMLoss, MaskedEdgeLoss
from utils import RunningAverage, colorize

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

torch.cuda.empty_cache()
import gc
gc.collect()
# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "DPTonGated-RGB"
# #logging = True
logging = False


def is_rank_zero(args):
    return args.rank == 0


import matplotlib


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################

    # model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)

    # load network
    # model_type = "dpt_large"
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
            scale=0.000040402,
            # shift=0.01,
            invert=True,
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
    ################################################################################################
    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
          experiment_name=args.name, optimizer_state_dict=None)

def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    PROJECT = args.name
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging
    if should_log:
        wandb.login(key="ee869683959b353d96c9594ddf667e1a7509df53")
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
        wandb.watch(model)

    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ###################################### losses ##############################################
    # criterion_ueff = SILogLoss()
    criterion_ueff = SILogLoss_l1()
    sms_loss = SMS_Loss()
    scale_invariant_loss = ScaleInvariantLoss()
    ssim_loss = DepthSSIMLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    edge_loss = MaskedEdgeLoss()
    ################################################################################################

    model.train()

    ###################################### Optimizer ################################################

    print("Using same LR")
    params = model.parameters()

    # for name, param in model.named_parameters():
    #     # print(name, param.size(), sep='\t')
    #     if 'pretrained' in name:  # 只训练 encoder 部分
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    # 选择优化器和损失函数，只优化 encoder 部分的参数
    # params = filter(lambda p: p.requires_grad, model.parameters())
    # for param in model.pretrained.parameters():
    #     nn.init.zeros_(param)

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr, betas=(0.9, 0.999))
    # optimizer = optim.AdamW([
    #     {'params': model.pos_embed.parameters(), 'weight_decay': 0.0},
    #     {'params': model.cls_token.parameters(), 'weight_decay': 0.0},
    #     {'params': model.norm.parameters(), 'weight_decay': 0.0},
    #     {'params': model.other_parameters(), 'weight_decay': 0.01}
    # ], lr=args.lr, betas=(0.9, 0.999))

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)

    ################################################################################################
    # max_iter = len(train_loader) * epochs
    l1_ratio = 1.0
    sms_ratio = 1.0
    sil_ratio = 10.0
    ssim_ratio = 10.0
    edge_ratio = 1.0
    for epoch in range(args.epoch, epochs):
        nanct = 0
        ################################# Train loop ##########################################################
        if should_log:
            wandb.log({"Epoch": epoch}, step=step)
            wandb.log({'Train/dpc rate': args.dpc_rate})
            wandb.log({'Train/Learning Rate': args.lr})
        lossPerEpoch = utils.RunningAverageDict()
        # train_loader = test_loader
        minmask = 100
        pbar = tqdm(enumerate(train_loader), desc=f"Epoch: {epoch}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(args) else enumerate(train_loader)
        for i, batch in pbar:
            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            rgb_depth = batch['rgb_depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            # print('0 img: ', img)
            # print('2 depth: ', depth)

            pred = model(img)
            pred = pred.unsqueeze(1)
            # print(pred)
            # zct = torch.sum((pred+0.00000001)<=0.0)
            # print(zct)

            mask = depth > args.min_depth
            rbg_depth_mask = rgb_depth > 1.0
            # print('3 pred: ', pred)
            # print('4 mask: ', mask)
            maskCount = torch.sum(mask)
            rbg_depth_mask_count = torch.sum(rbg_depth_mask)
            if maskCount < 10 or rbg_depth_mask_count < 10:
                print("\n  mask too less, id:", batch['id'][0],
                      "\n  depth mask:",maskCount.detach().cpu().numpy(),
                      "\n  rgb depth mask:",rbg_depth_mask_count.detach().cpu().numpy())
                step += 1
                continue
            # depth = depth - torch.min(depth[mask]) + 0.0003
            # depth = (1.0/depth)
            if epoch >40:
                l1_ratio = 0.8
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            # sms = sms_loss(pred, rgb_depth, mask=rbg_depth_mask.to(torch.bool), interpolate=True)
            sil = scale_invariant_loss(pred, rgb_depth, mask=rbg_depth_mask.to(torch.bool), interpolate=True)
            # ssim = ssim_loss(pred, rgb_depth, mask=rbg_depth_mask.to(torch.bool))
            edge = edge_loss(pred, depth, rbg_depth_mask.to(torch.bool))
            # print(l_dense)

            loss = (
                    l_dense * l1_ratio
                    # + sms_ratio * sms
                    + sil_ratio * sil
                    # + ssim_ratio * ssim
                    + edge * edge_ratio
                    )
            # 更新进度条的附加信息
            lossPerEpoch.update({'loss': loss.clone().cpu().detach().numpy(),
                                 'l_dense': l_dense.clone().cpu().detach().numpy() * l1_ratio,
                                 # 'sms': sms.clone().cpu().detach().numpy() * sms_ratio,
                                 'sil': sil.clone().cpu().detach().numpy() * sil_ratio,
                                 # 'ssim': ssim.clone().cpu().detach().numpy() * ssim_ratio,
                                 'edge': edge.clone().cpu().detach().numpy() * edge_ratio,
                                 })
            if torch.isnan(loss).any():
                nanct += 1
                step += 1
                print("Skipping the update due to NaN loss:", i)
                print(batch['id'])
                if nanct > 150:
                    print('breaken\n\n')
                    exit(0)
                continue  # 跳过这个 batch 的优化过程

            pbar.set_postfix(lossPerEpoch.get_value())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)

            step += 1
            scheduler.step()

            ########################################################################################################

            # if should_write and step % args.validate_every == 0:
            # if should_write and step % 1== 0:
            if should_write and step % (len(train_loader)) == 0:
                pbar.close()
                # print(minmask)

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)
                # print(metrics)
                # 创建并打开一个文件来追加写入，文件名为 data.txt
                rst_dir = os.path.join('checkpoints', run_id)
                os.makedirs(rst_dir, exist_ok=True)

                with open(os.path.join(rst_dir,'matrices.txt'), 'a') as file:
                    # 将转换好的字典字符串写入文件的一行
                    file.write(str(epoch) + str(metrics) + '\n')

                # print("Validated: {}".format(metrics))
                if should_log:
                    wandb.log({
                        f"Test/{criterion_ueff.name}": val_si.get_value(),
                        # f"Test/{criterion_bins.name}": val_bins.get_value()
                    }, step=step)

                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)

                if epoch%6 == 0.0:
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_epoch{epoch}.pt",
                                             root=rst_dir)

                if metrics['rmse'] < best_loss and should_write:
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_best.pt",
                                             root=rst_dir)
                    best_loss = metrics['rmse']
                model.train()
                print(experiment_name)
                #################################################################################################

        #     break
        # break

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            pred = model(img)
            pred = pred.unsqueeze(1)

            mask = depth > args.min_depth
            # depth = depth - torch.min(depth[mask]) + 0.0003
            # depth = (1.0 / depth)

            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())


            metrics.update(utils.compute_errors(args, depth, pred, mask))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':
    parser = allArgs.getArgs()

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        # print(torch.cuda.current_device)
        # print(torch.cuda.memory_summary(device=('gpu',0), abbreviated=False))
        main_worker(args.gpu, ngpus_per_node, args)
