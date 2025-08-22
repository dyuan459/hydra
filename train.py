# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
import argparse
import importlib
import time
import logging
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader

"""YOLOV3: Utility functions for YOLOv3 integration
These functions help manage the differences between classification and detection tasks
"""
def check_yolo_compatibility(args):
    """Verifies and updates arguments for YOLOv3 compatibility"""
    # Check for required arguments and set defaults if needed
    if not hasattr(args, 'num_workers'):
        args.num_workers = 4
    
    # Set task type for proper evaluation metrics
    args.task = "detection"
    
    # YOLOv3 specific settings
    if not hasattr(args, 'image_size'):
        args.image_size = 416  # Default YOLO image size
        
    # Add compatibility warning if using classification-specific arguments
    if args.arch != "yolov3" and "yolo" not in args.arch.lower():
        logging.warning(f"Using architecture '{args.arch}' with YOLOv3 code. This may cause compatibility issues.")
    
    return args

from YOLOv3 import load_model  # Custom import for YOLOv3 model loading
import models
import data
from args import parse_args
from utils.schedules import get_lr_policy, get_optimizer
from utils.logging import (
    save_checkpoint,
    create_subdirs,
    parse_configs_file,
    clone_results_to_latest_subdir,
)
from utils.semisup import get_semisup_dataloader
from utils.model import (
    get_layers,
    prepare_model,
    initialize_scaled_score,
    scale_rand_init,
    show_gradients,
    current_model_pruned_fraction,
    sanity_check_paramter_updates,
    snip_init,
)
from augmentations import StrongAug, TRANSFORM_VAL, TRANSFORM_TRAIN
import lossSingle

# TODO: update wrn, resnet models. Save both subnet and dense version.
# TODO: take care of BN, bias in pruning, support structured pruning
"""YOLOV3: Custom collate function for YOLO dataloader
This function handles both dictionary annotations from COCO and tensor annotations
from previous processing, ensuring consistent target format.
"""
def yolo_collate_fn(batch):
    imgs, targets = [], []
    for img, target in batch:
        imgs.append(img)
        
        # Check if target is already a tensor from previous transformations
        if isinstance(target, torch.Tensor):
            # Target is already a tensor, use it directly
            target_tensor = target
            
        # Check if target is a single dictionary - wrap in list for consistent handling
        elif isinstance(target, dict) and 'bbox' in target:
            boxes = []
            x, y, w, h = target['bbox']
            # Convert to YOLO format as with multiple annotations
            center_x = x + w/2
            center_y = y + h/2
            
            img_h, img_w = img.shape[1], img.shape[2]
            
            center_x /= img_w
            center_y /= img_h
            w /= img_w
            h /= img_h
            
            class_id = target.get('category_id', 0)
            if class_id > 0:
                class_id -= 1
                
            boxes.append([class_id, center_x, center_y, w, h])
            target_tensor = torch.tensor(boxes, dtype=torch.float32)
            
        # Assume target is a list of dictionaries (standard COCO format)
        elif isinstance(target, (list, tuple)):
            boxes = []
            for ann in target:
                # Skip if not a dictionary or doesn't have bbox
                if not isinstance(ann, dict):
                    continue
                    
                # Extract bounding box information
                if 'bbox' in ann:
                    # COCO format: [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    # Convert to center format used by YOLO: [center_x, center_y, width, height]
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Get image dimensions for normalization
                    img_h, img_w = img.shape[1], img.shape[2]
                    
                    # Normalize coordinates to [0, 1]
                    center_x /= img_w
                    center_y /= img_h
                    w /= img_w
                    h /= img_h
                    
                    # Class ID (COCO has 80 classes but indexed from 1, adjust to 0-indexed)
                    class_id = ann.get('category_id', 0)
                    if class_id > 0:  # Adjust for 0-indexing
                        class_id -= 1
                    
                    boxes.append([class_id, center_x, center_y, w, h])
            
            # Convert list to tensor if we have boxes, otherwise create empty tensor
            if boxes:
                target_tensor = torch.tensor(boxes, dtype=torch.float32)
            else:
                target_tensor = torch.zeros((0, 5), dtype=torch.float32)
        else:
            # Unknown format, create empty tensor
            print(f"Warning: Unknown target type in collate_fn: {type(target)}")
            target_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
            targets.append(target_tensor)

        # Stack all images into a batch
        if len(imgs) > 0:
            imgs = torch.stack(imgs, 0)
        else:
            imgs = torch.zeros((0, 3, 416, 416))  # Default size

        # Convert list of per-image target tensors into a single tensor
        # Expected format for lossSingle: each row is [img_idx, class, x, y, w, h]
        all_targets = []
        for img_idx, t in enumerate(targets):
            if not isinstance(t, torch.Tensor):
                continue
            if t.numel() == 0:
                continue
            # Ensure tensor is float
            t = t.float()
            # prepend image index column
            idx_col = torch.full((t.shape[0], 1), float(img_idx), dtype=t.dtype)
            # t currently: [class, cx, cy, w, h] -> desired: [img_idx, class, cx, cy, w, h]
            t = torch.cat((idx_col, t), dim=1)
            all_targets.append(t)

        if len(all_targets) > 0:
            targets = torch.cat(all_targets, dim=0)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        return imgs, targets

def main():
    args = parse_args()
    parse_configs_file(args)
    
    """YOLOV3: Ensure YOLOv3 compatibility with command-line arguments
    Check and update arguments to ensure they work correctly with YOLOv3
    """
    args = check_yolo_compatibility(args)

    # sanity checks
    if args.exp_mode in ["prune", "finetune"] and not args.resume:
        assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                n + 1,
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Select GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")

    # Create model
    # cl, ll = get_layers(args.layer_type) # simply gets the proper layers which is accounted for now
    # if len(gpu_list) > 1:
    #     print("Using multiple GPUs")
    #     model = nn.DataParallel(
    #         models.__dict__[args.arch](
    #             cl, ll, args.init_type, num_classes=args.num_classes
    #         ),
    #         gpu_list,
    #     ).to(device)
    # else:
    #     model = models.__dict__[args.arch](
    #         cl, ll, args.init_type, num_classes=args.num_classes
    #     ).to(device)
    """YOLOV3: Modified model loading to properly support YOLOv3 architecture
    This ensures proper initialization with the specified layer_type (dense/subnet)
    and properly loads pretrained weights if specified in source_net
    """
    #! TODO: You will have to adjust your model here for v5
    model = load_model("yolov3.cfg", args.layer_type)  # Load without weights first
    logger.info(model)
    
    # Set task type for proper evaluation metrics
    args.task = "detection"  # Add detection task type for evaluation

    # Customize models for training/pruning/fine-tuning
    prepare_model(model, args)
    # YOLOv3 models often carry COCO-specific anchors and buffers (e.g., grid
    # cell shapes, anchor priors). `prepare_model` must not accidentally
    # freeze/unfreeze these non-parameter buffers. If detection heads have
    # per-anchor scalars or other non-Parameter attributes, update
    # `prepare_model` in `utils/model.py` to treat them appropriately.

    # Setup tensorboard writer
    writer = SummaryWriter(os.path.join(result_sub_dir, "tensorboard"))

    """YOLOV3: Enhanced COCO dataset loading for YOLOv3
    Creates properly configured dataloaders for training and validation sets
    using custom transforms and collate functions specific to YOLOv3
    """
    # Set up paths - checking for existence and using proper transforms
    #! TODO: This is based on very scuffed paths, you will have to adjust them.
    train_images_path = "/home/danry/links/projects/def-rsolisob/danry/Efficient-Robust-Object-Detection/data/COCO2017/images/train_sample"
    train_annot_path = "/home/danry/links/projects/def-rsolisob/danry/Efficient-Robust-Object-Detection/data/COCO2017/annotations/instances_train2017_modified_sample.json"
    val_images_path = "/home/danry/links/projects/def-rsolisob/danry/Efficient-Robust-Object-Detection/data/COCO2017/images/valid_sample"
    val_annot_path = "/home/danry/links/projects/def-rsolisob/danry/Efficient-Robust-Object-Detection/data/COCO2017/annotations/instances_val2017_modified_sample.json"
    
    # Verify paths exist
    for path in [train_images_path, train_annot_path, val_images_path, val_annot_path]:
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
    
    # Create dataset objects with proper transforms
    # Preferred: use the `transforms=` argument available in newer torchvision
    # which expects a callable (image, target) -> (image, target).
    # Fallback: if older torchvision doesn't accept `transforms=`, use a small
    # subclass that applies our (img, target) callable in __getitem__.
    try:
        train_dataset = CocoDetection(
            root=train_images_path,
            annFile=train_annot_path,
            transforms=TRANSFORM_TRAIN,
        )

        test_dataset = CocoDetection(
            root=val_images_path,
            annFile=val_annot_path,
            transforms=TRANSFORM_VAL,
        )
    except TypeError:
        # Older torchvision: CocoDetection does not accept `transforms=` kwarg.
        # Define a minimal subclass that stores our callable and applies it
        # to (img, target) in __getitem__.
        class CocoDetectionWithTransforms(CocoDetection):
            def __init__(self, root, annFile, transforms=None):
                # Call parent with transform=None so parent returns raw (img, target)
                super(CocoDetectionWithTransforms, self).__init__(root, annFile, transform=None, target_transform=None)
                self.user_transforms = transforms

            def __getitem__(self, index):
                img, target = super(CocoDetectionWithTransforms, self).__getitem__(index)
                if self.user_transforms is not None:
                    return self.user_transforms(img, target)
                return img, target

        train_dataset = CocoDetectionWithTransforms(train_images_path, train_annot_path, transforms=TRANSFORM_TRAIN)
        test_dataset = CocoDetectionWithTransforms(val_images_path, val_annot_path, transforms=TRANSFORM_VAL)
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        collate_fn=yolo_collate_fn,
        pin_memory=(use_cuda and torch.cuda.is_available())
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        collate_fn=yolo_collate_fn,
        pin_memory=(use_cuda and torch.cuda.is_available())
    )

    # Log dataset sizes
    logger.info(f"Dataset: COCO, Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")

    # COCO is an object-detection dataset. Its dataloader returns images and
    # lists/dicts of detection targets (boxes, labels, areas, iscrowd, etc.).
    # The HYDRA repo's CIFAR/SVHN loaders are classification-focused and
    # return (images, class_labels). For COCO/YOLOv3 you must replace the
    # dataloader with a COCO-compatible loader that yields targets in the
    # format expected by your YOLOv3 implementation (typically a list of
    # dicts or tensors per image). Also adapt training loss (trainer) to use
    # YOLO detection loss rather than CrossEntropy/TRADES.

    # Semi-sup dataloader
    if args.is_semisup:
        # logger.info("Using semi-supervised training")
        # sm_loader = get_semisup_dataloader(args, D.tr_train) #TODO: how to semisup?
        pass
    else:
        sm_loader = None

    # autograd
    criterion = lossSingle.compute_loss
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    logger.info([criterion, optimizer, lr_policy])

    # train & val method
    trainer = importlib.import_module(f"trainer.{args.trainer}").train
    val = getattr(importlib.import_module("utils.eval"), args.val_method)

    """YOLOV3: Enhanced checkpoint loading to properly handle YOLOv3 model architecture
    This handles both standard PyTorch checkpoints (.pth) and darknet weights (.weights)
    It also properly deals with DataParallel prefixes and detects missing/unexpected keys
    """
    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info(f"Loading source net from {args.source_net}")
            
            # Handle different file formats
            if args.source_net.endswith(".weights"):
                # Load darknet weights format directly
                model.load_darknet_weights(args.source_net)
                logger.info(f"=> loaded darknet weights from '{args.source_net}'")
            else:
                # Load PyTorch checkpoint format
                checkpoint = torch.load(args.source_net, map_location=device)
                
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                
                # Handle potential module. prefix from DataParallel
                if all(k.startswith('module.') for k in state_dict.keys()):
                    from collections import OrderedDict
                    state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
                
                # Load with strict=False to see what's missing
                result = model.load_state_dict(state_dict, strict=False)
                
                if result.missing_keys:
                    logger.info(f"Missing keys: {result.missing_keys}")
                if result.unexpected_keys:
                    logger.info(f"Unexpected keys: {result.unexpected_keys}")
                
                logger.info(f"=> loaded checkpoint '{args.source_net}'")
        else:
            logger.info(f"=> no checkpoint found at '{args.source_net}'")
            
            # If no checkpoint found but source_net specified, use pretrained YOLOv3 weights
            if args.source_net == "pretrained":
                logger.info("Attempting to use pretrained YOLOv3 weights")
                model_path = "yolov3_ckpt_best.pth"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                    logger.info(f"=> loaded pretrained weights from '{model_path}'")
                else:
                    logger.info(f"=> no pretrained weights found at '{model_path}'")


    # Init scores once source net is loaded.
    # NOTE: scaled_init_scores will overwrite the scores in the pre-trained net.
    if args.scaled_score_init:
        initialize_scaled_score(model)

    # `--scaled-score-init` overwrites `popup_scores` from weight magnitudes.
    # For COCO-trained YOLOv3 you may prefer to initialize only backbone
    # layers' scores from weight magnitudes and leave detection heads
    # (anchors, output convolutions) with a targeted init to avoid hurting
    # detection quality.

    # Scaled random initialization. Useful when training a high sparse net from scratch.
    # If not used, a sparse net (without batch-norm) from scratch will not coverge.
    # With batch-norm its not really necessary.
    if args.scale_rand_init:
        scale_rand_init(model, args.k)

    # Scaled random initialization. Useful when training a high sparse net from scratch.
    # If not used, a sparse net (without batch-norm) from scratch will not coverge.
    # With batch-norm its not really necessary.
    if args.scale_rand_init:
        scale_rand_init(model, args.k)

    """YOLOV3: Enhanced SNIP initialization for YOLOv3
    SNIP initialization is modified to properly handle detection loss and targets,
    ensuring popup_scores are initialized based on detection sensitivity
    """
    if args.snip_init:
        logger.info("Performing SNIP initialization for YOLOv3 detection model")
        # For YOLO, we need to make sure the criterion takes model as third parameter
        # and handles the detection-specific target format
        try:
            # Get a small batch for initialization
            sample_images, sample_targets = next(iter(train_loader))
            sample_images = sample_images.to(device)
            
            # Customize the snip_init function call for detection
            snip_init(model, criterion, optimizer, train_loader, device, args)
            logger.info("SNIP initialization completed successfully")
        except Exception as e:
            logger.error(f"SNIP initialization failed: {str(e)}")
            logger.info("Falling back to standard initialization")

    assert not (args.source_net and args.resume), (
        "Incorrect setup: "
        "resume => required to resume a previous experiment (loads all parameters)|| "
        "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    )
    # resume (if checkpoint provided). Continue training with preiovus settings.
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Evaluate
    if args.evaluate or args.exp_mode in ["prune", "finetune"]:
        p1, _ = val(model, device, test_loader, criterion, args, writer)
        logger.info(f"Validation accuracy {args.val_method} for source-net: {p1}")
        if args.evaluate:
            return

    best_prec1 = 0

    show_gradients(model)

    if args.source_net:
        # Get state_dict directly since it was already processed when loading the model
        last_ckpt = model.state_dict()
    else:
        last_ckpt = copy.deepcopy(model.state_dict())

    # Start training
    for epoch in range(args.start_epoch, args.epochs + args.warmup_epochs):
        lr_policy(epoch)  # adjust learning rate

        # train
        trainer(
            model,
            device,
            train_loader,
            sm_loader,
            criterion,
            optimizer,
            epoch,
            args,
            writer,
        )

        """YOLOV3: Enhanced evaluation to properly handle object detection metrics
    For YOLOv3, we use mean Average Precision (mAP) as the primary evaluation metric
    instead of classification accuracy
    """
    # evaluate on test set
    if args.val_method == "smooth":
        prec1, radii = val(
            model, device, test_loader, criterion, args, writer, epoch
        )
        logger.info(f"Epoch {epoch}, mean provable Radii  {radii}")
    elif args.val_method == "mixtrain" and epoch <= args.schedule_length:
        prec1 = 0.0
    else:
        # For detection models like YOLOv3
        if args.task == "detection":
            # Use proper object detection evaluation
            mAP, class_APs = val(model, device, test_loader, criterion, args, writer, epoch)
            logger.info(f"Epoch {epoch}, mAP: {mAP:.4f}")
            # Log individual class APs if available
            if class_APs and len(class_APs) > 0:
                for cls_name, ap in class_APs.items():
                    logger.info(f"  Class {cls_name}: AP = {ap:.4f}")
            prec1 = mAP  # Use mAP as the metric for best model selection
        else:
            # Existing classification evaluation code
            prec1, _ = val(model, device, test_loader, criterion, args, writer, epoch)

        """YOLOV3: Enhanced checkpoint saving for YOLOv3
        For detection models, we save both the HYDRA format checkpoint and a YOLOv3-compatible format
        This ensures compatibility with standard YOLOv3 inference implementations
        """
        # remember best mAP and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        # Create checkpoint with additional YOLOv3 metadata
        checkpoint = {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,  # For detection, this is mAP
            "optimizer": optimizer.state_dict(),
            "task": args.task,  # Indicate this is a detection model
        }
        
        # Save checkpoint in HYDRA format
        save_checkpoint(
            checkpoint,
            is_best,
            args,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
            save_dense=args.save_dense,
        )
        
        # If requested to save dense model and this is the best model so far
        if args.save_dense and is_best:
            # Also save in standard YOLOv3 format for easier downstream use
            try:
                # Save darknet weights format for YOLO compatibility
                yolo_weights_path = os.path.join(result_sub_dir, "checkpoint", "model_best_yolov3.weights")
                model.save_darknet_weights(yolo_weights_path)
                logger.info(f"Saved YOLOv3-compatible weights to {yolo_weights_path}")
            except Exception as e:
                logger.error(f"Failed to save YOLOv3-compatible weights: {str(e)}")

        logger.info(
            f"Epoch {epoch}, val-method {args.val_method}, validation accuracy {prec1}, best_prec {best_prec1}"
        )
        if args.exp_mode in ["prune", "finetune"]:
            logger.info(
                "Pruned model: {:.2f}%".format(
                    current_model_pruned_fraction(
                        model, os.path.join(result_sub_dir, "checkpoint"), verbose=False
                    )
                )
            )
        # clone results to latest subdir (sync after every epoch)
        # Latest_subdir: stores results from latest run of an experiment.
        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )

        # Check what parameters got updated in the current epoch.
        sw, ss = sanity_check_paramter_updates(model, last_ckpt)
        logger.info(
            f"Sanity check (exp-mode: {args.exp_mode}): Weight update - {sw}, Scores update - {ss}"
        )

    current_model_pruned_fraction(
        model, os.path.join(result_sub_dir, "checkpoint"), verbose=True
    )


if __name__ == "__main__":
    main()
