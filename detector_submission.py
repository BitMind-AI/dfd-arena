# pm2 start detector_submission.py --no-autorestart -- \
#     --repo_id [huggingface model repo] \
#     --detector_module [detector registry module]

import os
import shutil
import argparse
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
import subprocess

def download_huggingface_model(repo_id, local_dir="submission_files"):
    """Downloads a Hugging Face model repository to a local directory."""
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"All files from '{repo_id}' downloaded to: {local_dir}")
    except Exception as e:
        return f"Failed: {e}"

def run_dfd_arena_with_pm2(detector_module):
    """
    Runs dfd_arena.py using pm2 with the specified detector module.
    
    Args:
    - detector_module (str): The name of the detector module to use.
    """
    # The command to run dfd_arena.py with pm2
    command = [
        "pm2", "start", "dfd_arena.py", 
        "--name", "dfd_arena",  # Naming the process
        "--",  # Passes subsequent arguments to the script
        "--detectors", detector_module
    ]
    
    try:
        # Run the command using subprocess
        subprocess.run(command, check=True)
        print(f"Successfully started dfd_arena.py with pm2 using detector module: {detector_module}")
    except subprocess.CalledProcessError as e:
        print(f"Error running dfd_arena.py with pm2: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model repo to a local directory")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="The Hugging Face model repo name (e.g., 'bert-base-uncased')")
    parser.add_argument("--detector_module", type=str, required=True,
                        help="The name of the registered detector module in DETECTOR_REGISTRY")
    parser.add_argument("--local_dir", type=str, default="submission_files",
                        help="The local directory to save files (default is 'submission_files')")
    args = parser.parse_args()
    args = parser.parse_args()
    download_huggingface_model(repo_id=args.repo_id, local_dir=args.local_dir)
    run_dfd_arena_with_pm2(detector_module=args.detector_module)

if __name__ == "__main__":
    main()