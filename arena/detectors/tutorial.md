# Tutorial: Adding a New Deepfake Detector

This tutorial will guide you through the process of adding a new deepfake detector to the DFD Arena framework. We'll use the SPSL (Spatial-Phase Shallow Learning) architecture, sourced from DeepfakeBench, as an example.

## 1. Get Your Model Weights

First, you need to have your model weights ready. This can be either your own architecture or a pretrained one from prior literature. For this example, we're using weights sourced from DeepfakeBench and hosted in a BitMind Hugging Face

## 2. Add Your Model-Specific Files to the Architectures Directory

Create a directory for your architecture within `arena/architectures/`. For our SPSL example, we've created [`arena/architectures/SPSL/`](../architectures/SPSL/).

## 3. Define Your Detector Class

Next, create a detector script within the `detectors` directory. For our SPSL example, we'll name it `spsl_detector.py`. In this new script, import all necessary dependencies and define your detector class. You may utilize the config parameter for model loading (see UCF for example usage).

For those interested in using the detector registry system, which is particularly useful for multi-model detectors like CAMO, you can refer to [`registry.py`](registry.py) and [`camo_detector.py`](camo_detector.py) as examples.

Here is an example of [`spsl_detector.py`](spsl_detector.py):
```python
from arena.detectors.registry import DETECTOR_REGISTRY
from arena.detectors.deepfake_detector import DeepfakeDetector
from arena.architectures.SPSL.config.constants import CONFIGS_DIR, WEIGHTS_DIR
from arena.architectures.SPSL.detectors import DETECTOR

@DETECTOR_REGISTRY.register_module(module_name='SPSL')
class SPSLDetector(DeepfakeDetector):
    def __init__(self, model_name: str = 'SPSL', config: str = 'spsl.yaml', device: str = 'cpu'):
        super().__init__(model_name, config, device)

    ...

    def infer(self, image_tensor):
        """ Perform inference using the model. """
        with torch.no_grad():
            pred_dict = self.model({'image': image_tensor})
        return pred_dict['prob']

    def __call__(self, image: Image) -> float:
        image_tensor = self.preprocess(image)
        return self.infer(image_tensor)
```

## 4. Import and Register Your Detector

Import your detector in the [`arena/detectors/__init__.py`](__init__.py) file:

```python
from .spsl_detector import SPSLDetector
```

## 5. Evaluate Your Detector

To evaluate your detector, you can use the existing evaluation scripts in the DFD Arena framework. Make sure your architecture is properly integrated and can be selected for evaluation using the configuration file you created.

## 6. Set Up Gates (if necessary)

If your architecture requires any specific gates or preprocessing steps, you can set them up in a similar manner to how you registered the architecture. Use the `GATE_REGISTRY` for this purpose.

## Understanding the Registration Process

The `DETECTOR_REGISTRY` is imported within `neurons/miner.py`. When you run `setup_miner_env.sh`, it generates a `miner.env` file that specifies which detector configuration to use. The miner script then loads this configuration, instantiates the corresponding architecture, and utilizes it within its forward function.

By following these steps, you've successfully added a new deepfake detector architecture to the DFD Arena framework. You can now use and evaluate your detector alongside other implemented models.