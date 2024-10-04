# Deepfake Detection Arena

An open-source framework for benchmarking deepfake detection models against AI-generated datasets with a wide variety of content.

## Background
The landscape of open-source computer vision currently grapples with a critical shortage of datasets and evaluation frameworks designed to benchmark systems that distinguish between real and AI-generated images. [Previous studies](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9721302) have predominantly targeted content-specific subsets of this problem, such as human faces in images and videos (e.g., [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)).

These efforts are valuable for testing new model architectures under limited conditions, but they do not adequately address the broad spectrum of image types encountered in everyday scenarios. 

**Deepfake Detection Arena (DFD-Arena)** aims to fill this gap by providing a comprehensive and adaptable benchmark suitable for the diverse and complex nature of in-the-wild images.

## Features

- **Benchmark Multiple Detectors**: Evaluate different deepfake detection models in a unified environment.
- **Support for Multiple Datasets**: Easily benchmark against various real and synthetic datasets.
- **Command-Line Customization**: Configure your benchmark run's name, output directory, detector models, and datasets at the CLI.
- **Result Persistence**: Save benchmarking results for future analysis.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
  - [Adding Datasets](#adding-datasets)
  - [Adding Detectors](#adding-detectors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation
### Clone the Repository

```bash
git clone git@github.com:BitMind-AI/dfd-arena.git
cd dfd-arena
```

### Instal System Dependencies
If you're benchmarking any detectors that use [dlib](http://dlib.net/), your system must have `cmake` installed.

```bash
./install_system_deps.sh
```

### Install Python Dependencies
We recommend setting up a Python virtual enviornment of your choice prior to this step. 

Instructions for setting up conda are available in the [miniconda quick command line install guide](https://docs.anaconda.com/miniconda/#quick-command-line-install).

With conda, you can create and activate your environment like this:

```bash
conda create -y -n arena python=3.10 ipython jupyter ipykernel
conda activate arena
```

With your virtual env activated, you can install `dfd-arena`:
```bash
pip install -r requirements.txt
```

## Usage
```
python dfd_arena.py 
```

You can customize your run with the following arguments:
```
python script.py --log-dir ./benchmark_runs --run-name my_benchmark --detectors CAMO UCF NPR --dataset-config arena/datasets.yaml
```

## Contributing

### Adding Datasets
*coming soon*

### Adding Detectors
*coming soon*

## License

This repository is licensed under the MIT License.

```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```

## Acknowledgements
Thank you to the authors of the Deepfake Bench ([paper](https://arxiv.org/abs/2307.01426), [repository](https://github.com/SCLBD/DeepfakeBench)), who provide a framework for training and evaluating models for detecting face deepfakes, and are also the authors behind the UCF model ([UCF: Uncovering Common Features for Generalizable Deepfake Detection](http://export.arxiv.org/abs/2304.13949)). 
