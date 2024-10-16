import unittest
from PIL import Image
import os
import sys
import numpy as np
import importlib
import argparse
from arena.detectors.registry import DETECTOR_REGISTRY

class TestDeepfakeDetector(unittest.TestCase):
    def setUp(self):
        """Set up the necessary information to test any DeepfakeDetector subclass."""
        self.script_dir = os.path.dirname(__file__)
        # Set the path of the sample image
        self.image_path = os.path.join(self.script_dir, 'sample_image.jpg')

        # Initialize the detector
        self.detector = DETECTOR_REGISTRY[self.detector_name]()
        self.detector.dataset_type = 'real'

    def test_inference(self):
        """Test model inference on a preprocessed image."""
        image = Image.open(self.image_path)
        prediction = self.detector(image)
        print(f"Prediction: {prediction}, Type: {type(prediction)}")
        self.assertIsInstance(prediction, np.ndarray, "Output should be a np.ndarray containing a float value")
        self.assertTrue(0 <= prediction <= 1, "Output should be between 0 and 1")

def update_dataset(detectors_repo_id, hf_token, detector_name, passed):
    """Update the Hugging Face dataset with the test results."""
    from datasets import load_dataset, Dataset
    # Load the dataset
    dataset = load_dataset(detectors_repo_id, split='train', token=hf_token)

    # Find the row with 'detector_name' equal to detector_name
    def update_row(example):
        if example['detector_name'] == detector_name:
            if passed: example['passed_invocation_test'] = "Passed"
            else: example['passed_invocation_test'] = "Failed"
        return example

    # Apply the update to the dataset
    updated_dataset = dataset.map(update_row)

    # Push changes back to the Hugging Face Hub
    updated_dataset.push_to_hub(detectors_repo_id, token=hf_token)
    print("Dataset updated successfully on Hugging Face Hub.")

if __name__ == '__main__':
    # Parse the command-line argument for the detector name
    parser = argparse.ArgumentParser(description="Run unit tests for a DeepfakeDetector subclass.")
    parser.add_argument('--detector_name', type=str, required=True, help="The base name of the detector (e.g., 'NPR' for 'NPRDetector')")
    parser.add_argument("--detectors_repo_id", type=str, required=True, help="The Hugging Face dataset repo name containing all submission details")
    parser.add_argument('--hf_token', type=str, default='', help='HuggingFace token used update the results dataset')
    args = parser.parse_args()

    # Pass the detector name as a parameter to the test suite
    TestDeepfakeDetector.detector_name = args.detector_name

    # Load the test suite
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDeepfakeDetector)

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the test suite
    result = runner.run(suite)

    # Check if all tests passed
    if result.wasSuccessful():
        print("Successful!")
        update_dataset(args.detectors_repo_id, args.hf_token, args.detector_name, passed=True)