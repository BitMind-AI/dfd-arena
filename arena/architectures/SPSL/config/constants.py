import os

# Path to the directory containing the constants.py file
CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))

# The base directory for related files
BASE_PATH = os.path.abspath(os.path.join(CONFIGS_DIR, ".."))
# Absolute paths for the required files and directories
CONFIG_PATH = os.path.join(CONFIGS_DIR, "spsl.yaml")  # Path to the .yaml file
WEIGHTS_DIR = os.path.join(BASE_PATH, "weights/") # Path to pretrained weights directory

HF_REPO = "bitmind/spsl"
BACKBONE_CKPT = "xception_best.pth"