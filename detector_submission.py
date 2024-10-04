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
        # Download the files to the local directory
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"All files from '{repo_id}' downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"Failed to download files: {e}")
        return None

def move_files(local_dir, detector_file, configs, module):
    """Move downloaded files to the appropriate directories."""
    base_dir = "arena/detectors/"
    detector_dir = os.path.join(base_dir, "deepfake_detectors/")
    configs_dir = os.path.join(detector_dir, "configs/")
    module_dir = os.path.join(base_dir, module)

    # Create directories if they don't exist
    os.makedirs(detector_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(module_dir, exist_ok=True)

    # Iterate over all files in the local directory and move them
    for file_name in os.listdir(local_dir):
        file_path = os.path.join(local_dir, file_name)
        try:
            if file_name == detector_file:
                # Move the detector file
                shutil.move(file_path, os.path.join(detector_dir, file_name))
                print(f"Moved {file_name} to {detector_dir}")
            elif file_name == configs:
                # Move the configs file
                shutil.move(file_path, os.path.join(configs_dir, file_name))
                print(f"Moved {file_name} to {configs_dir}")
            else:
                # Move the rest to a new directory
                shutil.move(file_path, os.path.join(module_dir, file_name))
                print(f"Moved {file_name} to {module_dir}")
        except Exception as e:
            print(f"Error moving files: {e}")
            
    shutil.rmtree(local_dir)
    print(f"Deleted the original directory: {local_dir}")

def run_dfd_arena_with_pm2(detector_module):
    """
    Runs dfd_arena.py using pm2 with the specified detector module.
    
    Args:
    - detector_module (str): The name of the detector module to use.
    """
    # The command to run dfd_arena.py with pm2
    command = [
        "pm2", "start", "dfd_arena.py", 
        "--no-autorestart",
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
    parser.add_argument("--detector_file", type=str, required=True,
                        help="The name of the .py file containing the DeepfakeDetector subclass")
    parser.add_argument("--configs", type=str, required=True,
                        help="The name of the .YAML configs for the DeepfakeDetector")
    parser.add_argument("--module", type=str, required=True,
                        help="The name of the registered detector module in DETECTOR_REGISTRY")
    parser.add_argument("--local_dir", type=str, default="submission_files",
                        help="The local directory to save files (default is 'submission_files')")
    args = parser.parse_args()
    args = parser.parse_args()
    
    downloaded_dir = download_huggingface_model(repo_id=args.repo_id, local_dir=args.local_dir)
    if downloaded_dir:
        move_files(downloaded_dir, args.detector_file, args.configs, args.module)
    else:
        print("Failed to download files.")
    run_dfd_arena_with_pm2(detector_module=args.module)

if __name__ == "__main__":
    main()