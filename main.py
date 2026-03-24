# Copyright (c) EEEM071, University of Surrey

import datetime
import math
import os
import os.path as osp
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from src import models
from src.data_manager import ImageDataManager
from src.eval_metrics import evaluate
from src.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from src.lr_schedulers import init_lr_scheduler
from src.optimizers import init_optimizer
from src.utils.avgmeter import AverageMeter
from src.utils.experiment_logger import ExperimentLogger
from src.utils.generaltools import set_random_seed
from src.utils.iotools import check_isfile
from src.utils.loggers import Logger, RankLogger
from src.utils.torchtools import (
    count_num_param,
    accuracy,
    load_pretrained_weights,
    save_checkpoint,
    resume_from_checkpoint,
)
from src.utils.visualtools import visualize_ranked_results
import uuid

def run(args, redirect_stdout=True):
    """
    Main entry point for a single training / evaluation run.

    Parameters
    ----------
    args            : parsed argparse Namespace
    redirect_stdout : when True (CLI default), redirects stdout to a log file.
                      Set to False when calling from a Jupyter notebook so that
                      output appears in the cell rather than being written to disk.
    """
    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    if redirect_stdout:
        log_name = "log_test.txt" if args.evaluate else "log_train.txt"
        sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========")
    student_id = os.environ.get('STUDENT_ID', '<your id>')
    student_name = os.environ.get('STUDENT_NAME', '<your name>')
    print("Student ID:{}".format(student_id))
    print("Student name:{}".format(student_name))
    print("UUID:{}".format(uuid.uuid4()))
    print("Experiment time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print("==========")
    print(f"==========\nArgs:{args}\n==========")
    print(f"[data_fraction={args.data_fraction}] Using {args.data_fraction * 100:.1f}% of the dataset.")

    if use_gpu:
        print(f"Currently using GPU {args.gpu_devices}")
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print(f"Initializing model: {args.arch}")
    model = models.init_model(
        name=args.arch,
        num_classes=dm.num_train_pids,
        loss={"xent", "htri"},
        pretrained=not args.no_pretrained,
        use_gpu=use_gpu,
    )
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    criterion_xent = CrossEntropyLoss(
        num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth
    )
    criterion_htri = TripletLoss(margin=args.margin)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(
            args.resume, model, optimizer=optimizer
        )

    logger = ExperimentLogger(args.save_dir, vars(args))

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            print(f"Evaluating {name} ...")
            queryloader = testloader_dict[name]["query"]
            galleryloader = testloader_dict[name]["gallery"]
            result = test(model, queryloader, galleryloader, use_gpu, args=args)

            is_best = logger.log_eval(0, {
                "val_mAP":    result["mAP"],
                "val_rank1":  result["rank1"],
                "val_rank5":  result["rank5"],
                "val_rank10": result["rank10"],
                "val_rank20": result["rank20"],
                "val_mINP":   result["mINP"],
            })

            if args.visualize_ranks:
                visualize_ranked_results(
                    test(model, queryloader, galleryloader, use_gpu, return_distmat=True, args=args),
                    dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, "ranked_results", name),
                    topk=20,
                )

        logger.generate_plots()
        logger.write_summary()
        logger.print_final_report(args.save_dir)
        logger.close()
        return

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print("=> Start training")

    try:
        for epoch in range(args.start_epoch, args.max_epoch):
            train_metrics = train(
                epoch,
                model,
                criterion_xent,
                criterion_htri,
                optimizer,
                trainloader,
                use_gpu,
                args,
            )
            logger.log_train_epoch(epoch + 1, train_metrics)

            scheduler.step()

            if (
                (epoch + 1) > args.start_eval
                and args.eval_freq > 0
                and (epoch + 1) % args.eval_freq == 0
                or (epoch + 1) == args.max_epoch
            ):
                print("=> Test")

                for name in args.target_names:
                    print(f"Evaluating {name} ...")
                    queryloader = testloader_dict[name]["query"]
                    galleryloader = testloader_dict[name]["gallery"]
                    result = test(model, queryloader, galleryloader, use_gpu, args=args)
                    ranklogger.write(name, epoch + 1, result["rank1"])

                    is_best = logger.log_eval(epoch + 1, {
                        "val_mAP":    result["mAP"],
                        "val_rank1":  result["rank1"],
                        "val_rank5":  result["rank5"],
                        "val_rank10": result["rank10"],
                        "val_rank20": result["rank20"],
                        "val_mINP":   result["mINP"],
                    })

                ckpt_path = osp.join(args.save_dir, "checkpoint_ep{}.pth.tar".format(epoch + 1))
                save_checkpoint(
                    {
                        "state_dict": model.state_dict(),
                        "rank1": result["rank1"],
                        "mAP":   result["mAP"],
                        "epoch": epoch + 1,
                        "arch":  args.arch,
                        "optimizer": optimizer.state_dict(),
                    },
                    args.save_dir,
                    is_best=is_best,
                )
                if is_best:
                    logger.set_best_ckpt(osp.join(args.save_dir, "best_model.pth.tar"))

    finally:
        elapsed = round(time.time() - time_start)
        elapsed_str = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed {elapsed_str}")
        ranklogger.show_summary()

        logger.generate_plots()
        logger.write_summary()
        logger.print_final_report(args.save_dir)
        logger.close()


def train(
    epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, args
):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    grad_norms = AverageMeter()

    model.train()
    for p in model.parameters():
        p.requires_grad = True  # open all layers

    epoch_start = time.time()
    end = time.time()

    pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1:>3} train", unit="batch", leave=False)
    for batch_idx, (imgs, pids, _, _) in enumerate(pbar):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs, features = model(imgs)
        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            xent_loss = criterion_xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        optimizer.zero_grad()
        loss.backward()

        # NaN/Inf guard
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                if not torch.isfinite(param_norm):
                    raise RuntimeError(
                        f"Non-finite gradient at epoch {epoch + 1}, batch {batch_idx + 1}. "
                        "Consider lowering the learning rate or checking data."
                    )
                total_norm += param_norm.item() ** 2
        total_norm = math.sqrt(total_norm)
        grad_norms.update(total_norm)

        optimizer.step()

        batch_time.update(time.time() - end)

        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])

        # get current LR (first param group)
        cur_lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix(
            loss=f"{(xent_losses.avg + htri_losses.avg):.3f}",
            xent=f"{xent_losses.avg:.3f}",
            tri=f"{htri_losses.avg:.3f}",
            acc=f"{accs.avg:.1f}%",
            lr=f"{cur_lr:.1e}",
        )

        if (batch_idx + 1) % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.4f} ({data_time.avg:.4f})\t"
                "Xent {xent.val:.4f} ({xent.avg:.4f})\t"
                "Htri {htri.val:.4f} ({htri.avg:.4f})\t"
                "Acc {acc.val:.2f} ({acc.avg:.2f})\t".format(
                    epoch + 1,
                    batch_idx + 1,
                    len(trainloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    xent=xent_losses,
                    htri=htri_losses,
                    acc=accs,
                )
            )

        end = time.time()

    pbar.close()
    epoch_time = time.time() - epoch_start
    total_samples = len(trainloader.dataset) if hasattr(trainloader, "dataset") else 0
    sps = total_samples / epoch_time if epoch_time > 0 else 0.0

    gpu_mem_mb = None
    if use_gpu and torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    cur_lr = optimizer.param_groups[0]["lr"]

    return {
        "total_loss":         xent_losses.avg + htri_losses.avg,
        "id_loss":            xent_losses.avg,
        "triplet_loss":       htri_losses.avg,
        "train_accuracy":     accs.avg,
        "learning_rate":      cur_lr,
        "grad_norm":          grad_norms.avg,
        "epoch_time":         epoch_time,
        "samples_per_second": sps,
        "gpu_memory_mb":      gpu_mem_mb,
    }


def test(
    model,
    queryloader,
    galleryloader,
    use_gpu,
    ranks=[1, 5, 10, 20],
    return_distmat=False,
    args=None,
):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for imgs, pids, camids, _ in tqdm(queryloader, desc="Extracting query features", leave=False):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print(
            "Extracted features for query set, obtained {}-by-{} matrix".format(
                qf.size(0), qf.size(1)
            )
        )

        gf, g_pids, g_camids = [], [], []
        for imgs, pids, camids, _ in tqdm(galleryloader, desc="Extracting gallery features", leave=False):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print(
            "Extracted features for gallery set, obtained {}-by-{} matrix".format(
                gf.size(0), gf.size(1)
            )
        )

    test_batch_size = args.test_batch_size if args is not None else "?"
    print(f"=> BatchTime(s)/BatchSize(img): {batch_time.avg:.3f}/{test_batch_size}")

    m, n = qf.size(0), gf.size(0)
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    if return_distmat:
        return distmat

    print("Computing CMC and mAP")
    cmc, mAP, mINP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print(f"mAP: {mAP:.1%}")
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print(f"mINP: {mINP:.1%}")
    print("------------------")

    return {
        "rank1":  cmc[0],
        "rank5":  cmc[4] if len(cmc) > 4 else None,
        "rank10": cmc[9] if len(cmc) > 9 else None,
        "rank20": cmc[19] if len(cmc) > 19 else None,
        "mAP":    mAP,
        "mINP":   mINP,
    }


def main():
    """CLI entry point — parses argv and calls run()."""
    parser = argument_parser()
    args = parser.parse_args()
    run(args, redirect_stdout=True)


if __name__ == "__main__":
    main()
