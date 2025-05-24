from ultralytics import YOLO

model = YOLO("/mnt/base/YOLO/Fire-Detection/runs/detect/train2/weights/best.pt")

model.info()

# Export the model
model.export(format="onnx")