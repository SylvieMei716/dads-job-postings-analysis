# import os
# from ultralytics import YOLO
# import yaml
# import logging

# def clean_labels(label_dir):
#     for label_file in os.listdir(label_dir):
#         label_path = os.path.join(label_dir, label_file)
#         with open(label_path, 'r') as file:
#             lines = file.readlines()
        
#         cleaned_lines = []
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) == 5:  # YOLO format: class x_center y_center width height
#                 cleaned_lines.append(line)
        
#         with open(label_path, 'w') as file:
#             file.writelines(cleaned_lines)

# # Load paths from data.yaml
# with open('data.yaml', 'r') as file:
#     data_config = yaml.safe_load(file)

# # Define paths to your label directories
# train_label_dir = os.path.abspath(data_config['train_labels'])
# val_label_dir = os.path.abspath(data_config['val_labels'])
# test_label_dir = os.path.abspath(data_config['test_labels'])

# # Clean the label directories
# clean_labels(train_label_dir)
# clean_labels(val_label_dir)
# clean_labels(test_label_dir)

# print("Labels cleaned successfully.")

# # Setup logging
# logging.basicConfig(filename='evaluation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# # Function to log precision, recall, and other metrics every 100 steps
# def custom_log_metrics(results, log_type="Validation"):
#     total_steps = len(results['precision'])  # Assuming 'precision' length represents the number of steps
#     for step in range(0, total_steps, 100):
#         precision_at_step = results['precision'][step]
#         recall_at_step = results['recall'][step]
#         logging.info(f"{log_type} - Step {step}: Precision = {precision_at_step}, Recall = {recall_at_step}")

#     # Log final metrics after the evaluation is complete
#     final_precision = results['precision'][-1]
#     final_recall = results['recall'][-1]
#     logging.info(f"{log_type} - Final Precision = {final_precision}, Final Recall = {final_recall}")

# # Check for an existing checkpoint (pt file) to resume training
# checkpoint_path = "yolo11n.pt"  # Update this with your checkpoint file's path
# if os.path.exists(checkpoint_path):
#     print(f"Resuming training from checkpoint: {checkpoint_path}")
#     model = YOLO(checkpoint_path)  # Load from checkpoint
# else:
#     print("No checkpoint found, loading pretrained model from scratch.")
#     model = YOLO("yolo11n.pt")  # Load from pretrained model

# # Train the model (will continue from the loaded checkpoint if available)
# results = model.train(data="data.yaml", epochs=3)

# # Evaluate the model's performance on the validation set
# val_results = model.val()

# # Log precision and recall every 100 steps for validation
# custom_log_metrics(val_results, log_type="Validation")

# # Apply the model to the test set and evaluate performance
# print("Evaluating on the test set...")
# test_results = model.val(data="data.yaml", split='test')  # Use test data for evaluation

# # Log precision and recall every 100 steps for test evaluation
# custom_log_metrics(test_results, log_type="Test")

# # Export the model to ONNX format
# success = model.export(format="onnx")

# print("Model training, evaluation, and test evaluation completed successfully.")

import os
from ultralytics import YOLO
import yaml
import logging

# Setup logging for evaluation
logging.basicConfig(filename='evaluation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to clean the labels directory (YOLO format)
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


# Function to log precision, recall, and other metrics every 100 steps
def custom_log_metrics(results, log_type="Validation"):
    total_steps = len(results['metrics']['precision'])  # Assuming 'precision' is the metric
    for step in range(0, total_steps, 100):
        precision_at_step = results['metrics']['precision'][step]
        recall_at_step = results['metrics']['recall'][step]
        logging.info(f"{log_type} - Step {step}: Precision = {precision_at_step}, Recall = {recall_at_step}")

    # Log final metrics after the evaluation is complete
    final_precision = results['metrics']['precision'][-1]
    final_recall = results['metrics']['recall'][-1]
    logging.info(f"{log_type} - Final Precision = {final_precision}, Final Recall = {final_recall}")


# Function to train the model
def train_model(data_yaml, epochs=3, pretrained_model="yolov8n.pt"):
    print("Starting training process...")
    # Load the pretrained YOLO model
    model = YOLO(pretrained_model)

    # Train the model
    results = model.train(data=data_yaml, epochs=epochs)

    # After training, clean up memory by deleting the model object
    del model
    print("Training completed and model object removed from memory.")

    return results


# Function to evaluate the model (validation/test)
def evaluate_model(data_yaml, split='val', batch_size=1, pretrained_model="yolov8n.pt"):
    print(f"Starting evaluation process on {split} set...")
    
    # Load the pretrained YOLO model
    model = YOLO(pretrained_model)

    # Evaluate the model on the specified split
    eval_results = model.val(data=data_yaml, split=split, batch=batch_size)

    # Log precision and recall every 100 steps for the evaluation
    custom_log_metrics(eval_results, log_type=split.capitalize())

    # Clean up memory after evaluation by deleting the model object
    del model
    print(f"Evaluation on {split} set completed and model object removed from memory.")

    return eval_results


# Function to export the model to ONNX format
def export_model(pretrained_model="yolov8n.pt"):
    # Load the pretrained YOLO model
    model = YOLO(pretrained_model)

    # Export the model to ONNX format
    success = model.export(format="onnx")

    # Clean up memory after exporting
    del model
    print("Model exported to ONNX format and model object removed from memory.")
    return success


# Main pipeline
if __name__ == "__main__":
    # Load paths from data.yaml
    with open('data.yaml', 'r') as file:
        data_config = yaml.safe_load(file)

    # Clean the label directories
    clean_labels(os.path.abspath(data_config['train_labels']))
    clean_labels(os.path.abspath(data_config['val_labels']))
    clean_labels(os.path.abspath(data_config['test_labels']))

    print("Labels cleaned successfully.")

    # Train the model
    train_model('data.yaml', epochs=3)

    # Evaluate the model on the validation set
    evaluate_model('data.yaml', split='val')

    # Evaluate the model on the test set
    evaluate_model('data.yaml', split='test')

    # Export the model to ONNX format
    export_model()