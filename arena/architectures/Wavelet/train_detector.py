"""
SPSL Trainer Module

This script is an adaptation of the DeepfakeBench training code, originally authored by Zhiyuan Yan
(zhiyuanyan@link.cuhk.edu.cn) and further modified for the SPSL (Spatial-Phase Shallow Learning) architecture.

Key differences from train.py:
1. Simplified dataset handling: Uses HuggingFace datasets instead of custom dataset classes.
2. Streamlined configuration: Combines multiple config files into a single YAML.
3. Enhanced logging: Includes more detailed timing and progress information.
4. Modular design: Separates concerns into distinct functions for better maintainability.
5. Specific to SPSL: Includes SPSL-specific configurations and model initialization.

The script handles the entire training pipeline including:
- Argument parsing
- Configuration loading
- Dataset preparation
- Model initialization
- Training loop execution
- Evaluation
- Logging and checkpointing
"""
import os
from os.path import join
import random
import gc
import argparse
import yaml
from pathlib import Path
import time
import datetime
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torchvision import transforms

from huggingface_hub import hf_hub_download
from logger import create_logger, RankFilter
from metrics.utils import parse_metric_for_print

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from bitmind.utils.data import load_and_split_datasets, create_real_fake_datasets
from bitmind.image_transforms import base_transforms, random_aug_transforms, CLAHE, ConvertToRGB, CenterCrop
from bitmind.constants import DATASET_META, TARGET_IMAGE_SIZE

from detectors import DETECTOR
from trainer.trainer import Trainer

# Constants for file paths and configurations
from config.constants import (
    CONFIG_PATH,
    WEIGHTS_DIR,
    HF_REPO,
    BACKBONE_CKPT
)

# Set up command-line argument parser
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, default=CONFIG_PATH, help='path to detector YAML file')
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--workers', type=int, default=os.cpu_count() - 1,
                    help='number of workers for data loading')
parser.add_argument('--epochs', type=int, default=None, help='number of training epochs')

args = parser.parse_args()
torch.cuda.set_device(args.local_rank)

if torch.cuda.is_available():
    print("CUDA is available!")

def ensure_backbone_is_available(logger,
                                 weights_dir=WEIGHTS_DIR,
                                 model_filename=BACKBONE_CKPT,
                                 hugging_face_repo_name=HF_REPO):
    """
    Ensures that the backbone model weights are available locally.
    If not, downloads them from the specified HuggingFace repository.

    Args:
        logger: Logger object for logging messages
        weights_dir (str): Directory to store weights
        model_filename (str): Filename of the model weights
        hugging_face_repo_name (str): HuggingFace repository name

    Returns:
        None
    """
    destination_path = Path(weights_dir) / Path(model_filename)
    if not destination_path.parent.exists():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory {destination_path.parent}.")
    if not destination_path.exists():
        model_path = hf_hub_download(hugging_face_repo_name, model_filename)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        torch.save(model, destination_path)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Downloaded backbone {model_filename} to {destination_path}.")
    else:
        logger.info(f"{model_filename} backbone already present at {destination_path}.")
    
    return str(destination_path)

def init_seed(config):
    """
    Initializes random seeds for reproducibility.

    Args:
        config (dict): Configuration dictionary containing seed information

    Returns:
        None
    """
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])

def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Processes a batch of data and returns a dictionary with tensors.

    Args:
        batch (list): List of tuples containing (image, label)

    Returns:
        dict: Dictionary containing batched tensors
    """
    images, labels = zip(*batch)
    
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels) 
    
    data_dict = {
        'image': images,
        'label': labels,
    }    
    return data_dict

def prepare_datasets(config, logger):
    """
    Prepares datasets for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary
        logger: Logger object for logging messages

    Returns:
        tuple: Train, validation, and test data loaders
    """
    start_time = log_start_time(logger, "Loading and splitting individual datasets")
    
    fake_datasets = load_and_split_datasets(config['dataset_meta']['fake'])
    real_datasets = load_and_split_datasets(config['dataset_meta']['real'])

    log_finish_time(logger, "Loading and splitting individual datasets", start_time)
    
    start_time = log_start_time(logger, "Creating real fake dataset splits")
    train_dataset, val_dataset, test_dataset = \
    create_real_fake_datasets(real_datasets,
                              fake_datasets,
                              config['split_transforms']['train'],
                              config['split_transforms']['validation'],
                              config['split_transforms']['test'],
                              source_labels=False)

    log_finish_time(logger, "Creating real fake dataset splits", start_time)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['train_batchSize'],
                                               shuffle=True,
                                               num_workers=config['workers'],
                                               drop_last=True,
                                               collate_fn=custom_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['train_batchSize'],
                                             shuffle=True,
                                             num_workers=config['workers'],
                                             drop_last=True,
                                             collate_fn=custom_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config['train_batchSize'],
                                              shuffle=True, 
                                              num_workers=config['workers'],
                                              drop_last=True,
                                              collate_fn=custom_collate_fn)

    logger.info(f"Train size: {len(train_loader.dataset)}")
    logger.info(f"Validation size: {len(val_loader.dataset)}")
    logger.info(f"Test size: {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader

def choose_optimizer(model, config):
    """
    Selects and initializes the optimizer based on the configuration.

    Args:
        model: The model to optimize
        config (dict): Configuration dictionary containing optimizer settings

    Returns:
        optimizer: Initialized optimizer
    """
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
    else:
        raise NotImplementedError(f'Optimizer {opt_name} is not implemented')
    return optimizer

def choose_scheduler(config, optimizer):
    """
    Selects and initializes the learning rate scheduler based on the configuration.

    Args:
        config (dict): Configuration dictionary containing scheduler settings
        optimizer: The optimizer to schedule

    Returns:
        scheduler: Initialized learning rate scheduler
    """
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError(f'Scheduler {config["lr_scheduler"]} is not implemented')
    return scheduler

def choose_metric(config):
    """
    Selects the evaluation metric based on the configuration.

    Args:
        config (dict): Configuration dictionary containing metric settings

    Returns:
        str: Name of the chosen metric
    """
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError(f'metric {metric_scoring} is not implemented')
    return metric_scoring

def log_start_time(logger, process_name):
    """Log the start time of a process."""
    start_time = time.time()
    logger.info(f"{process_name} Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    return start_time

def log_finish_time(logger, process_name, start_time):
    """Log the finish time and elapsed time of a process."""
    finish_time = time.time()
    elapsed_time = finish_time - start_time

    # Convert elapsed time into hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Log the finish time and elapsed time
    logger.info(f"{process_name} Finish Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_time))}")
    logger.info(f"{process_name} Elapsed Time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

def save_config(config, outputs_dir):
    """
    Saves a config dictionary as both a pickle file and a YAML file, ensuring only basic types are saved.
    Also, lists like 'mean' and 'std' are saved in flow style (on a single line).
    
    Args:
        config (dict): The configuration dictionary to save.
        outputs_dir (str): The directory path where the files will be saved.
    """

def main():
    """
    Main function to run the training pipeline.
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load configurations
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(os.getcwd() + '/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    config.update(config2)

    config['workers'] = args.workers
    
    config['local_rank'] = args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat'] = False

    if args.epochs:
        config['nEpochs'] = args.epochs
    
    spsl_transforms = transforms.Compose([
        ConvertToRGB(),
        CenterCrop(),
        transforms.Resize(TARGET_IMAGE_SIZE),
        CLAHE(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    config['split_transforms'] = {
        'train': spsl_transforms,
        'validation': spsl_transforms,
        'test': spsl_transforms
    }
    
    config['dataset_meta'] = DATASET_META
    dataset_names = [item["path"] for datasets in config['dataset_meta'].values() for item in datasets]
    config['train_dataset'] = dataset_names
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    
    # Create logger
    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    outputs_dir = os.path.join(config['log_dir'], config['model_name'] + '_' + timenow)
    os.makedirs(outputs_dir, exist_ok=True)
    logger = create_logger(os.path.join(outputs_dir, 'training.log'))
    config['log_dir'] = outputs_dir
    logger.info(f'Save log to {outputs_dir}')
    
    config['ddp'] = args.ddp

    # Initialize seed
    init_seed(config)

    # Set cudnn benchmark
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))


    backbone_path = ensure_backbone_is_available(logger=logger,
                                 model_filename=config['pretrained'].split('/')[-1],
                                 hugging_face_repo_name='bitmind/' + config['model_name'])
    
    config['pretrained'] = backbone_path
    
    # Prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    
    # Prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # Prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # Prepare the metric
    metric_scoring = choose_metric(config)

    # Prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

    # Prepare the data loaders
    train_loader, val_loader, test_loader = prepare_datasets(config, logger)

    # Print and save configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += f"{key}: {value}\n"
    logger.info(params_string)
    save_config(config, outputs_dir)
    
    # Start training
    start_time = log_start_time(logger, "Training")
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch,
                    train_data_loader=train_loader,
                    validation_data_loaders={'val': val_loader}
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with validation {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info(f"Stop Training on best Validation metric {parse_metric_for_print(best_metric)}") 
    log_finish_time(logger, "Training", start_time)
   
    # Test
    start_time = log_start_time(logger, "Test")
    trainer.eval(eval_data_loaders={'test': test_loader}, eval_stage="test")
    log_finish_time(logger, "Test", start_time)
    
    # Update scheduler
    if scheduler is not None:
        scheduler.step()

    # Close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()


