from ultralytics import YOLO
from openvino.inference_engine import IECore
ie = IECore()

# 选择硬件设备
net = ie.read_network(model="openvino_model.xml", weights="openvino_model.bin")
exec_net = ie.load_network(network=net, device_name="CPU")  # 或 "GPU", "MYRIAD", "FPGA"

# Load a YOLOv8n PyTorch model
model = YOLO(r"E:\yolov10\yolov10-main\models\x200.pt")

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'
# Load the exported OpenVINO model
ov_model = YOLO(r"E:\yolov10\yolov10-main\models\x200_openvino_model")



# Run inference
results = ov_model(r"E:\BackGround\2022329600041-3.jpg")
