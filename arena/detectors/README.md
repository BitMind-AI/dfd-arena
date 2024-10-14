# Tutorial: Adding a New Deepfake Detector

This tutorial will guide you through the process of adding a new deepfake detector to the DFD Arena framework. We'll use the SPSL (Spatial-Phase Shallow Learning) detector as an example.

## 1. Get Your Model

First, you need to have your model ready. This can be either your own architecture or a pretrained one from prior literature. For this example, we're using the SPSL model.

## 2. Add Your Model to a Subdirectory

Create a new subdirectory within `arena/detectors/` for your model. For SPSL, we have:

## 3. Create a Script for Your Model

Create a new Python script for your model within the `arena/detectors/deepfake_detectors/` directory. For our SPSL example, we'll create `spsl_detector.py`.

## 4. Define and Register Your Detector Class

In your new script (e.g., `spsl_detector.py`), define your detector class and register it using the `DETECTOR_REGISTRY`. Here's an example:

[`dfd-arena/arena/detectors/deepfake_detectors/spsl_detector.py`](spsl_detector.py)
```
from arena.detectors.registry import DETECTOR_REGISTRY
@DETECTOR_REGISTRY.register_module(module_name='SPSL')
class SPSLDetector:
def init(self, config):
# Initialize your model here
pass
def forward(self, x):
# Implement the forward pass of your model
pass
```

## 5. Import Your Detector
Import your detector in the [`arena/detectors/deepfake_detectors/__init__.py`](__init__.py):
```
from .spsl_detector import SPSLDetector
```
This step ensures that your detector is available when the framework loads detectors.

## 6. Evaluate Your Detector

To evaluate your detector, you can use the existing evaluation scripts in the DFD Arena framework. Make sure your detector is properly integrated and can be selected for evaluation.

## 7. Set Up Gates (if necessary)

If your detector requires any specific gates or preprocessing steps, you can set them up in a similar manner to how you registered the detector. Use the `GATE_REGISTRY` for this purpose.

## Understanding the Registration Process

The `DETECTOR_REGISTRY` is imported within `neurons/miner.py`. When you run `setup_miner_env.sh`, it generates a `miner.env` file that specifies which detector to use. The miner script then loads this detector and utilizes it within its forward function.

By following these steps, you've successfully added a new deepfake detector to the DFD Arena framework. You can now use and evaluate your detector alongside other implemented models.