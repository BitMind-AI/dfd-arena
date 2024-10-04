from .deepfake_detector import DeepfakeDetector
import os
import importlib
import inspect

# Dynamically import all .py files in the directory
current_dir = os.path.dirname(__file__)
module_names = [
    filename[:-3] for filename in os.listdir(current_dir)
    if filename.endswith('.py') and filename != '__init__.py'
]

# Import each module and look for DeepfakeDetector subclasses
for module_name in module_names:
    module = importlib.import_module(f'.{module_name}', package=__name__)
    
    # Inspect the module for DeepfakeDetector subclasses
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, DeepfakeDetector) and obj is not DeepfakeDetector:
            # Optionally: Add to globals to make available in package namespace
            globals()[name] = obj