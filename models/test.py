from model_controller import ModelController
import os

try:
    # Use the same model path as run.py
    model_path = "./models/walking_micro.onnx"
    if not os.path.isfile(model_path):
        print(f"Model file not found at {model_path}")
    else:
        self.model_controller = ModelController(model_path)
except Exception as e:
    print(f"Warning: Could not load walking model: {e}")
