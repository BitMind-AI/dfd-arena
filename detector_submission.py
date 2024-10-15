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

def run_dfd_arena_with_pm2(detector_module, results_repo_id, detectors_repo_id, hf_token, ec2_automation=False):
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
            "--leaderboard-submission", "True",
            "--detectors", detector_module,
            "--results-repo-id", results_repo_id,
            "--detectors-repo-id", detectors_repo_id,
            "--hf-token", hf_token
        ]
    else:
        command = [
            "pm2", "start", "dfd_arena.py", 
            "--no-autorestart",
            "--name", "dfd_arena",  # Naming the process
            "--",  # Passes subsequent arguments to the script
            "--leaderboard-submission", "True",
            "--detectors", detector_module,
            "--results-repo-id", results_repo_id,
            "--detectors-repo-id", detectors_repo_id,
            "--hf-token", hf_token
        ]
    
    try:
        # Run the command using subprocess
        subprocess.run(command, check=True)
        print(f"Successfully started dfd_arena.py with pm2 using detector module: {detector_module}")
    except subprocess.CalledProcessError as e:
        print(f"Error running dfd_arena.py with pm2: {e}")

def run_detector_test_with_pm2(detector_name, detectors_repo_id, hf_token):
    """
    Runs the unit test script (test_detector.py) using pm2 with the specified detector name.
    Returns True if the tests passed, False otherwise.
    """
    test_file_path = "arena/detectors/deepfake_detectors/unit_tests/test_detector.py"
    command = [
        "pm2", "start", test_file_path,
        "--no-autorestart",
        "--name", f"test_{detector_name}_detector",  # Naming the process
        "--",  # Passes subsequent arguments to the script
        "--detector_name", detector_name,
        "--detectors_repo_id", detectors_repo_id,
        "--hf_token", hf_token
    ]
    
    try:
        # Run the command using subprocess
        subprocess.run(command, check=True)
        print(f"Successfully started {test_file_path} with pm2 using detector: {detector_name}")
        # Check if the process finished successfully
        result = subprocess.run(['pm2', 'status', f'test_{detector_name}_detector'], capture_output=True, text=True)
        if "errored" in result.stdout.lower():
            print(f"Unit test failed for {detector_name}.")
            return False
        else:
            print(f"Unit test passed for {detector_name}.")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {test_file_path} with pm2: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model repo to a local directory")
    parser.add_argument("--detector_name", type=str, required=True,
                        help="The detector name to evaluate")
    parser.add_argument("--detectors_repo_id", type=str, required=True,
                        help="The Hugging Face dataset repo name containing all submission details")
    parser.add_argument("--local_dir", type=str, default="submission_files",
                        help="The local directory to save files (default is 'submission_files')")
    parser.add_argument('--results_repo_id', type=str, default='',
                        help='Path to leaderboard results dataset repo on HuggingFace')
    parser.add_argument('--hf_token', type=str, default='', help='HuggingFace token used update the results dataset')
    parser.add_argument('--ec2_automation', action='store_true', default=False)
    args = parser.parse_args()
    args = parser.parse_args()
    dataset = load_dataset(args.detectors_repo_id)['train']
    filtered_row = dataset.filter(lambda x: x['detector_name'] == args.detector_name)
    
    # Check if the 'evaluation_status' in the filtered row is 'Benchmarking'
    if len(filtered_row) > 0:
        evaluation_status = filtered_row[0]['evaluation_status']
        print(f"The evaluation status of {args.detector_name} is {evaluation_status}.")
        if evaluation_status != 'Benchmarking':
            print('Exiting: evaluation_status != Benchmarking')
            sys.exit(1)
    else:
        print(f"Exiting: No row found with 'detector_name' equal to {args.detector_name}.")
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

    if filtered_dataset['passed_invocation_test'] == 'Pending':
        test_passed = run_detector_test_with_pm2(args.detector_name, args.detectors_repo_id, args.hf_token)
        if not test_passed:
            print("Detector invocation unit tests failed. Not running dfd_arena_with_pm2.")
            sys.exit(1)
        print("Detector invocation unit tests passed. Running dfd_arena_with_pm2.")
        
    if filtered_dataset['passed_invocation_test'] == 'True':
        run_dfd_arena_with_pm2(detector_module=args.detector_name,
                               results_repo_id=args.results_repo_id,
                               detectors_repo_id=args.detectors_repo_id,
                               hf_token=args.hf_token,
                               ec2_automation=args.ec2_automation)

if __name__ == "__main__":
    main()


