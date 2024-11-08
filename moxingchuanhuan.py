from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO(r"E:\yolov10\yolov10-main\2022337621048\2022337621048.pt")

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

