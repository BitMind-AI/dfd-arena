"""
Class for the WaveletDetector.

This module implements the Wavelet-Packet detector
for face forgery detection in the frequency domain.

The WaveletDetector utilizes wavelet packet decomposition to capture
multi-scale frequency information from input images. This approach
allows for a more comprehensive analysis of the image's frequency
content, potentially revealing subtle artifacts introduced by
face forgery techniques.

Functions in the Class:
1. __init__: Initialization of the detector
2. build_backbone: Constructs the backbone network
3. build_loss: Builds the loss function
4. features: Extracts features from input data and wavelet representation
5. classifier: Performs classification on extracted features
6. get_losses: Computes losses for training
7. get_train_metrics: Calculates metrics during training
8. get_test_metrics: Calculates metrics during testing
9. forward: Performs forward pass, including wavelet packet computation
10. compute_wavelet_packet: Computes wavelet packet representation of input images

The wavelet packet decomposition is performed in the compute_wavelet_packet
method, which transforms the input image into a multi-scale frequency
representation. This representation is then concatenated with the original
image features for improved forgery detection.

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
from utils import wavelet_math

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='wavelet')
class WaveletDetector(AbstractDetector):
    """
    Wavelet-Packet detector for face forgery detection.

    This detector leverages wavelet packet decomposition to extract
    rich frequency domain features from input images. These features
    are combined with spatial domain features to enhance forgery detection.
    """

    def __init__(self, config):
        """
        Initialize the Wavelet-Packet detector.

        Args:
            config (dict): Configuration dictionary for the detector.
                           Should include parameters for wavelet packet computation
                           such as 'wavelet', 'max_lev', 'log_scale', 'mode', and 'cuda'.
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
            
        Note:
            This method adapts the pretrained 3-channel (RGB) conv1 layer to a
            4-channel input (RGB + wavelet) by averaging the pretrained weights
            across RGB channels and repeating for the 4th channel. This
            preserves pretrained knowledge while accommodating the
            additional wavelet information channel.
        """
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)

        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}

        # Create a new conv1 layer with 6 input channels (3 for image, 3 for wavelet)
        backbone.conv1 = nn.Conv2d(6, 32, 3, 2, 0, bias=False)
        
        # Check if 'conv1.weight' exists in the original state_dict
        if 'backbone.conv1.weight' in state_dict:
            conv1_data = state_dict['backbone.conv1.weight']
            # average across the RGB channels
            avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)
            # repeat the averaged weights across the 4 new channels
            backbone.conv1.weight.data = avg_conv1_data.repeat(1, 6, 1, 1)            
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
    
    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify the extracted features.

        Args:
            features (torch.Tensor): Combined spatial and wavelet features.

        Returns:
            torch.Tensor: Classification output.
        """
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """
        Compute losses for the detector.

        Args:
            data_dict (dict): Input data dictionary containing 'label'.
            pred_dict (dict): Prediction dictionary containing 'cls'.

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
            data_dict (dict): Input data dictionary containing 'label'.
            pred_dict (dict): Prediction dictionary containing 'cls'.

        Returns:
            dict: Dictionary of computed metrics (accuracy, AUC, EER, AP).
        """
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        self.video_names = []
        return metric_batch_dict
    
    def compute_wavelet_packet(self, img):
        """
        Compute wavelet packet representation of the input image.

        This method performs a wavelet packet decomposition on the input image
        using the Daubechies 1 (db1) wavelet. The decomposition is performed
        up to 3 levels, analyzing the image at different scales and frequencies.

        Mathematical process:
        1. Filtering: The image is convolved with low-pass (L) and high-pass (H) filters.
        2. Downsampling: Results are downsampled by a factor of 2.
        3. Recursion: This process is applied to all subbands up to level 3.

        For a 1D signal x[n], each level of decomposition can be expressed as:
        - Low-pass: y_L[n] = Σ_k h[k] * x[2n - k]
        - High-pass: y_H[n] = Σ_k g[k] * x[2n - k]
        Where h[k] and g[k] are the low-pass and high-pass filter coefficients.

        For 2D images, this process is applied separately to rows and columns.

        Args:
            img (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Wavelet packet representation of shape (B, C, H, W),
                        where the channel dimension contains aggregated 
                        frequency information from all subbands.

        Data flow:
        1. Reshape input from (B, C, H, W) to (B, H, W, C)
        2. Perform wavelet packet decomposition
        3. Aggregate features by averaging across subbands
        4. Reshape and interpolate to match original image dimensions

        Note:
        - The use of wavelet packets allows for a richer representation of 
        frequency content compared to standard wavelet transforms.
        - This method potentially captures subtle artifacts introduced by 
        face forgery techniques across multiple scales and frequencies.
        """
        b, c, h, w = img.shape
        
        # Convert from (B, C, H, W) to (B, H, W, C)
        img = img.permute(0, 2, 3, 1)
        
        # Convert to numpy array
        #img_np = img.cpu().numpy()
        
        # Compute wavelet packet representation
        wavelet_packets = wavelet_math.batch_packet_preprocessing(
            #img_np,
            img,
            wavelet='db1',
            max_lev=3,
            log_scale=False,
            mode='reflect',
            #cuda=True
        )
        
        # Convert back to torch tensor
        #wavelet_packets = torch.from_numpy(wavelet_packets).to(img.device)
        
        # Reshape from (B, N, H', W', C) to (B, C, H', W')
        wavelet_packets = wavelet_packets.mean(dim=1).permute(0, 3, 1, 2)
        
        # Resize to match original image dimensions
        wavelet_packets = F.interpolate(wavelet_packets, size=(h, w), mode='bilinear', align_corners=False)
        return wavelet_packets
    
    def features(self, data_dict: dict, wavelet_fea) -> torch.Tensor:
        """
        Extract features from the input data and wavelet representation.

        Args:
            data_dict (dict): Input data dictionary containing the 'image' key.
            wavelet_fea (torch.Tensor): Wavelet packet features.

        Returns:
            torch.Tensor: Combined features from spatial and frequency domains.
        """
        # Ensure wavelet_fea has the same number of channels as the input image
        features = torch.cat((data_dict['image'], wavelet_fea), dim=1)
        return self.backbone.features(features)
    
    def forward(self, data_dict: dict) -> dict:
        """
        Forward pass of the detector.

        This method performs the following steps:
        1. Compute wavelet packet representation of the input image
        2. Extract features from the combined spatial and wavelet data
        3. Perform classification on the extracted features

        Args:
            data_dict (dict): Input data dictionary containing 'image'.

        Returns:
            dict: Dictionary containing prediction results ('cls', 'prob', 'feat').
        """
        wavelet_fea = self.compute_wavelet_packet(data_dict['image'])
        features = self.features(data_dict, wavelet_fea)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict


    

