"""
Class for the SPSLDetector.

This module implements the Spatial-Phase Shallow Learning (SPSL) detector
for face forgery detection in the frequency domain.

Author: Zhiyuan Yan
Email: zhiyuanyan@link.cuhk.edu.cn
Date: 2023-07-06

Functions in the Class:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{liu2021spatial,
  title={Spatial-phase shallow learning: rethinking face forgery detection in frequency domain},
  author={Liu, Honggu and Li, Xiaodan and Zhou, Wenbo and Chen, Yuefeng and He, Yuan and Xue, Hui and Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={772--781},
  year={2021}
}

Note:
To ensure consistency in the comparison with other detectors, we have opted not to utilize
the shallow Xception architecture. Instead, we are employing the original Xception model.
"""

import os
import datetime
import logging
from typing import Union
from collections import defaultdict

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='spsl')
class SpslDetector(AbstractDetector):
    """
    Spatial-Phase Shallow Learning (SPSL) detector for face forgery detection.
    """

    def __init__(self, config):
        """
        Initialize the SPSL detector.

        Args:
            config (dict): Configuration dictionary for the detector.
        """
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        """
        Build the backbone network for the detector.

        Args:
            config (dict): Configuration dictionary for the backbone.

        Returns:
            nn.Module: The constructed backbone network.
        """
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)

        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}

        # Create a new conv1 layer with 4 input channels
        backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        # Check if 'conv1.weight' exists in the original state_dict
        if 'backbone.conv1.weight' in state_dict:
            conv1_data = state_dict['backbone.conv1.weight']
            # average across the RGB channels
            avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)
            # repeat the averaged weights across the 4 new channels
            backbone.conv1.weight.data = avg_conv1_data.repeat(1, 4, 1, 1)
            bt.logging.info('Initialized conv1 weights from pretrained model')
        else:
            bt.logging.info('Using random initialization for conv1')
            
        return backbone

    def build_loss(self, config):
        """
        Build the loss function for the detector.

        Args:
            config (dict): Configuration dictionary for the loss function.

        Returns:
            nn.Module: The constructed loss function.
        """
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict, phase_fea) -> torch.Tensor:
        """
        Extract features from the input data.

        Args:
            data_dict (dict): Input data dictionary.
            phase_fea (torch.Tensor): Phase features.

        Returns:
            torch.Tensor: Extracted features.
        """
        features = torch.cat((data_dict['image'], phase_fea), dim=1)
        return self.backbone.features(features)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify the extracted features.

        Args:
            features (torch.Tensor): Extracted features.

        Returns:
            torch.Tensor: Classification output.
        """
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """
        Compute losses for the detector.

        Args:
            data_dict (dict): Input data dictionary.
            pred_dict (dict): Prediction dictionary.

        Returns:
            dict: Dictionary of computed losses.
        """
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        """
        Compute training metrics for the detector.

        Args:
            data_dict (dict): Input data dictionary.
            pred_dict (dict): Prediction dictionary.

        Returns:
            dict: Dictionary of computed metrics.
        """
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        self.video_names = []
        return metric_batch_dict
    
    def forward(self, data_dict: dict) -> dict:
        """
        Forward pass of the detector.

        Args:
            data_dict (dict): Input data dictionary.

        Returns:
            dict: Dictionary containing prediction results.
        """
        phase_fea = self.phase_without_amplitude(data_dict['image'])
        features = self.features(data_dict, phase_fea)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict

    def phase_without_amplitude(self, img):
        """
        Extract phase information without amplitude from the input image.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Reconstructed image using phase information.
        """
        # Convert to grayscale
        gray_img = torch.mean(img, dim=1, keepdim=True)
        # Compute the DFT of the input signal
        X = torch.fft.fftn(gray_img, dim=(-1, -2))
        # Extract the phase information from the DFT
        phase_spectrum = torch.angle(X)
        # Create a new complex spectrum with the phase information and zero magnitude
        reconstructed_X = torch.exp(1j * phase_spectrum)
        # Use the IDFT to obtain the reconstructed signal
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X, dim=(-1, -2)))
        return reconstructed_x
