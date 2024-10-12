from collections import defaultdict
import random
import joblib
import time
import os
import argparse
import yaml

from arena.utils.metrics import compute_metrics
from arena.utils.data import load_datasets, upper_left_quadrant
from arena.detectors.registry import DETECTOR_REGISTRY


class Arena:
    """Class for benchmarking deepfake detection models."""

    def __init__(
        self,
        detectors,
        datasets,
        name=None,
        log_dir='.',
        leaderboard_submission=False,
        model_repo_id=None,
        detectors_repo_id=None,
        hf_token=None
    ):
        """
        Args:
            detectors (list): Names of detectors to benchmark.
            datasets (dict): Dictionary with keys 'path' for a huggingface dataset path
                             and 'split' for the dataset split to use.
            name (str, optional): Name for the benchmark. Defaults to None.
            log_dir (str): Directory where all pkl files will be logged.
        """
        self.name = name or f'benchmark-{time.time()}'
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.dataset_paths = [
            ds['path'] for ds in datasets['real'] + datasets['fake']
        ]
        real_datasets, fake_datasets = load_datasets(datasets)
        self.fake_datasets = fake_datasets
        self.real_datasets = real_datasets
        self.detectors = detectors

        self.metrics = {
            detector_name: {
                dataset_path: defaultdict(int)
                for dataset_path in self.dataset_paths + ['total']
            } for detector_name in detectors
        }
        self.dataset_indices = defaultdict(list)
        self.n_failed = defaultdict(int)
        self.leaderboard_submission = leaderboard_submission
        self.model_repo_id = model_repo_id
        self.detectors_repo_id = detectors_repo_id
        self.hf_token = hf_token

    def run_benchmarks(self):
        """Run benchmarks for all detectors against real and synthetic datasets."""
        for detector_name in self.detectors:
            print(f"Evaluating {detector_name}...")
            detector = DETECTOR_REGISTRY[detector_name]()
            detector.name = detector_name
            self.benchmark_detector(detector)
            print(f'----- Detector: {detector_name} -----')
            self.display_metrics(self.metrics[detector_name])
            if self.leaderboard_submission:
                self.push_result_to_hf(self.metrics[detector_name])
                self.push_detectors_update_to_hf(detector_name)
        self.save_results()

    def benchmark_detector(self, detector, n_dataset_samples=1000):
        """
        Benchmark a single detector against all datasets.

        Args:
            detector: The detector to benchmark.
            n_dataset_samples (int): Number of samples to use from each dataset.
        """
        # Infer on real datasets, track true negatives/false positives
        for dataset in self.real_datasets:
            print(f"Evaluating on real dataset {dataset}")
            self.evaluate_detector(detector, dataset, 'real', n_dataset_samples)

        # Infer on fake datasets, track false negatives/true positives
        for dataset in self.fake_datasets:
            print(f"Evaluating on fake dataset {dataset}")
            self.evaluate_detector(detector, dataset, 'fake', n_dataset_samples)

    def evaluate_detector(self, detector, dataset, dataset_type, n_dataset_samples):
        """
        Evaluate a detector on a single dataset.

        Args:
            detector: The detector to evaluate.
            dataset: The dataset to evaluate on.
            dataset_type (str): Either 'real' or 'fake'.
            n_dataset_samples (int): Number of samples to use from the dataset.
        """
        ds_path = dataset.huggingface_dataset_path
        sampling_new_indices = False
        if self.dataset_indices[ds_path]:
            indices = self.dataset_indices[ds_path]
        else:
            sampling_new_indices = True
            indices = random.sample(range(len(dataset)), n_dataset_samples)

        try:
            for detector_type in detector.detectors:
                detector.detectors[detector_type].dataset_type = dataset_type
        except AttributeError:
            detector.dataset_type = dataset_type

        for images_processed, image_idx in enumerate(indices):
            if images_processed >= n_dataset_samples + self.n_failed[ds_path]:
                break

            image = dataset[image_idx]['image']
            if image is None:
                self.n_failed[ds_path] += 1
                continue

            if sampling_new_indices:
                self.dataset_indices[ds_path].append(image_idx)

            pred = detector(image)
            if pred <= 0.5:
                key = 'tn' if dataset_type == 'real' else 'fn'
            else:
                key = 'tp' if dataset_type == 'fake' else 'fp'

            self.metrics[detector.name][ds_path][key] += 1
            self.metrics[detector.name]['total'][key] += 1

    def display_metrics(self, metrics):
        """
        Display metrics for each dataset and the total.

        Args:
            metrics (dict): Dictionary containing metrics for each dataset and total.
        """
        for ds_path, ds_metrics in metrics.items():
            print(ds_path)
            computed_metrics = compute_metrics(**ds_metrics)
            if ds_path == 'total':
                for metric, value in computed_metrics.items():
                    print(f'\t{metric}: {value}')
            else:
                print(f'\taccuracy: {computed_metrics["accuracy"]}')

    def save_results(self):
        """Save benchmark results to files."""
        output_dir = os.path.join(self.log_dir, self.name)
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.metrics, os.path.join(output_dir, 'metrics.pkl'))
        joblib.dump(self.dataset_indices, os.path.join(output_dir, 'dataset_indices.pkl'))
        joblib.dump(self.n_failed, os.path.join(output_dir, 'n_failed.pkl'))
        
    def push_result_to_hf(self, metrics):
        from datasets import load_dataset, Dataset
        from huggingface_hub import upload_file
        import pandas as pd
        
        results_dict = {}
        for ds_path, ds_metrics in metrics.items():
            computed_metrics = compute_metrics(**ds_metrics)
            if ds_path == 'total':
                for metric, value in computed_metrics.items():
                    results_dict[metric] = value
            else:
                results_dict[ds_path] = computed_metrics['accuracy']
                
        print(f"results_dict: {results_dict}")
        df_result = pd.DataFrame([results_dict])
        df_result['Detector'] = self.detectors[0]
        try:
            existing_dataset = load_dataset(self.model_repo_id, use_auth_token=self.hf_token)['train']
            df_existing = pd.DataFrame(existing_dataset)
            print(f"Existing dataset loaded: {df_existing}")
            df_updated = pd.concat([df_existing, df_result])
        except Exception as e:
            print(f"Dataset not found or unable to download: {str(e)}. Creating a new dataset.")
            df_updated = df_result

        print(f"df_updated: {df_updated}")
        # Save the updated (or new) DataFrame as a CSV file locally
        submission_file = "updated_results.csv"
        df_updated.to_csv(submission_file, index=False)
        
        try:
            dataset = load_dataset("csv", data_files=submission_file)
            dataset.push_to_hub(repo_id=self.model_repo_id, token=self.hf_token)
            return "Submission successful!"
        except Exception as e:
            return f"Failed to push submission: {str(e)}"
            
    def push_detectors_update_to_hf(self, detector_name):
        from huggingface_hub import upload_file, hf_hub_download
        import pandas as pd
        # Try to download the existing dataset
        existing_dataset_path = hf_hub_download(
            repo_id=self.detectors_repo_id,
            filename="submissions/submission.csv",
            token=self.hf_token,
            repo_type="dataset"
        )
        # If the file is found, load the existing dataset into a DataFrame
        df_updated = pd.read_csv(existing_dataset_path)
        df_updated.loc[df_updated['detector_name'] == detector_name, 'evaluation_progress'] = 'Complete'
        # Save the updated (or new) DataFrame as a CSV file
        submission_file = "submission.csv"
        df_updated.to_csv(submission_file, index=False)
        
        # Upload the updated (or new) file to the Hugging Face repository
        try:
            upload_file(
                path_or_fileobj=submission_file,
                path_in_repo="submissions/submission.csv",  # Location in the repo
                repo_id=self.detectors_repo_id,
                token=self.hf_token,
                repo_type="dataset"
            )
            return "Submission successful!"
        except Exception as e:
            return f"Failed to push submission: {str(e)}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark deepfake detection models.')
    parser.add_argument('--log-dir', type=str, default='./benchmark_runs', help='Directory where all pkl files will be logged')
    parser.add_argument('--run-name', type=str, default=None, help='Name of benchmarking run')
    parser.add_argument('--detectors', nargs='+', default=['CAMO', 'UCF', 'NPR'], help='List of detector names')
    parser.add_argument('--dataset-config', type=str, default='arena/datasets.yaml',
                        help='Path to YAML file containing datasets')
    parser.add_argument('--leaderboard-submission', type=bool, default='False',
                        help='Used to push the metrics dict to the HF results dataset when scoring a leaderboard submission')
    parser.add_argument('--model_repo_id', type=str, default='', help='Path to leaderboard results dataset repo on HuggingFace')
    parser.add_argument('--detectors_repo_id', type=str, default='', help='Path to leaderboard results dataset repo on HuggingFace')
    parser.add_argument('--hf_token', type=str, default='', help='HuggingFace token used update the results dataset')

    args = parser.parse_args()

    with open(args.dataset_config, 'r') as file:
        benchmark_datasets = yaml.safe_load(file)

    start = time.time()
    arena = Arena(
        detectors=args.detectors,
        datasets=benchmark_datasets,
        name=args.run_name,
        log_dir=args.log_dir,
        leaderboard_submission=args.leaderboard_submission,
        model_repo_id=args.model_repo_id,
        detectors_repo_id=args.detectors_repo_id,
        hf_token=args.hf_token,
    )
    arena.run_benchmarks()
    print(f"----- Finished benchmarking in {time.time() - start:.4f}s -----")
