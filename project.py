import os
from ultralytics import YOLO
import yaml

def clean_labels(label_dir):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as file:
            lines = file.readlines()
        
        cleaned_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:  # YOLO format: class x_center y_center width height
                cleaned_lines.append(line)
        
        with open(label_path, 'w') as file:
            file.writelines(cleaned_lines)

# Load paths from data.yaml
with open('data.yaml', 'r') as file:
    data_config = yaml.safe_load(file)

# Define paths to your label directories
train_label_dir = os.path.abspath(data_config['train_labels'])
val_label_dir = os.path.abspath(data_config['val_labels'])
test_label_dir = os.path.abspath(data_config['test_labels'])

# Clean the label directories
clean_labels(train_label_dir)
clean_labels(val_label_dir)
clean_labels(test_label_dir)

print("Labels cleaned successfully.")

# Train a YOLO model
# Set environment variable to avoid OpenMP runtime error
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# # Create a new YOLO model from scratch
# model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="data.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format="onnx")

print("Model training completed successfully.")