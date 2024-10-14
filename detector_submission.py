# pm2 start detector_submission.py --no-autorestart -- \
#      --detectors_repo_id caliangandrew/dfd-arena-detectors
#      --results_repo_id caliangandrew/dfd-arena-results \
#      --hf_token [token]

import os
import sys
import shutil
import argparse
from huggingface_hub import HfApi, snapshot_download
from datasets import load_dataset
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

def run_dfd_arena_with_pm2(detector_module, model_repo_id, detectors_repo_id, hf_token, ec2_automation=True):
    """
    Runs dfd_arena.py using pm2 with the specified detector module.
    
    Args:
    - detector_module (str): The name of the detector module to use.
    """
    if ec2_automation:
        command = [
            "pm2", "start", "dfd_arena.py", 
            "--no-autorestart",
            "--name", "dfd_arena",  # Naming the process
            "--output", "/home/ubuntu/detector_submission_out.log",  # Output log file
            "--error", "/home/ubuntu/detector_submission_err.log",   # Error log file
            "--",  # Passes subsequent arguments to the script
            "--leaderboard_submission", "True",
            "--detectors", detector_module,
            "--model_repo_id", model_repo_id,
            "--detectors_repo_id", detectors_repo_id,
            "--hf_token", hf_token
        ]
    else:
        command = [
            "pm2", "start", "dfd_arena.py", 
            "--no-autorestart",
            "--name", "dfd_arena",  # Naming the process
            "--",  # Passes subsequent arguments to the script
            "--leaderboard_submission", "True",
            "--detectors", detector_module,
            "--model_repo_id", model_repo_id,
            "--detectors_repo_id", detectors_repo_id,
            "--hf_token", hf_token
        ]
    
    try:
        # Run the command using subprocess
        subprocess.run(command, check=True)
        print(f"Successfully started dfd_arena.py with pm2 using detector module: {detector_module}")
    except subprocess.CalledProcessError as e:
        print(f"Error running dfd_arena.py with pm2: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model repo to a local directory")
    parser.add_argument("--detectors_repo_id", type=str, required=True,
                        help="The Hugging Face dataset repo name containing all submission details")
    parser.add_argument("--local_dir", type=str, default="submission_files",
                        help="The local directory to save files (default is 'submission_files')")
    parser.add_argument('--results_repo_id', type=str, default='',
                        help='Path to leaderboard results dataset repo on HuggingFace')
    parser.add_argument('--hf_token', type=str, default='', help='HuggingFace token used update the results dataset')
    args = parser.parse_args()
    args = parser.parse_args()
    dataset = load_dataset(args.detectors_repo_id)['train']
    filtered_dataset = dataset.filter(lambda x: x['evaluation_progress'] == 'Pending')
    if len(filtered_dataset) == 0:
        print("No rows with 'evaluation_progress' equal to 'Pending'. Exiting.")
        sys.exit(1)
    filtered_dataset = filtered_dataset[0]
    print(filtered_dataset)
    downloaded_dir = download_huggingface_model(repo_id=filtered_dataset["model_repo"], local_dir=args.local_dir)
    if downloaded_dir:
        move_files(downloaded_dir,
                   filtered_dataset["detector_file_path"],
                   filtered_dataset["configs_file_path"],
                   filtered_dataset["detector_name"])
    else:
        print("Failed to download files.")
    run_dfd_arena_with_pm2(detector_module=filtered_dataset["detector_name"],
                           model_repo_id=filtered_dataset["model_repo"],
                           detectors_repo_id=args.detectors_repo_id,
                           hf_token=args.hf_token)

if __name__ == "__main__":
    main()
