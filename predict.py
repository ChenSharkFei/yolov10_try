from sympy.physics.units import length

from ultralytics import YOLOv10

# Load a pretrained YoLov10n model
model = YOLOv10(r"E:\yolov10\yolov10-main\2022337621048\2022337621048.pt")
# Perform object detection on an image
#  results = model("test1.jpg")
results = model.predict(r"E:\yolov11\ultralytics-main\data\val\images\2022337621290-bz84.jpg")

results[0].show()



