
#Fire Detection using YOLOv9


##Project Overview
```
This project implements a fire detection system using the YOLOv9 object detection model. It supports training on custom fire datasets, real-time detection with webcam or images, and optimized versions for Raspberry Pi.
```




##Repository Structure
```text
Fire-Detection-YOLOv9/
│
├── Dataset # Annotated Dataset
├── models/ # Model weights (YOLOv9 full and tiny)
├── src/ # Source code for detection scripts and utilities
├── results/ # Evaluation results and prediction visuals
├── requirements.txt # Dependencies
├── README.md # Usage
├── .gitignore # Files to be ignored by git
└── LICENSE # Project license
```





#Getting Started



##Prerequisites
```bash
- Python 3.8+
- Pip package manager
```
##Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd Fire-Detection-YOLOv9
```
Install Dependencies:

```bash
pip install -r requirements.txt
```








## Usage

### Training the Model

You can train the YOLOv9 model on your custom fire dataset using the following Python code snippet or command line:

```python
from ultralytics import YOLO

# Load pretrained YOLOv9 model (YOLOv9-tiny for Raspberry Pi, or YOLOv9c for full)
model = YOLO("yolov9t.pt")

# Train on your dataset with specified parameters
results = model.train(data="./data.yaml", epochs=100, imgsz=640, device=[0,1])
```


Or from the command line (assuming you have a training script):

```bash
python src/train.py --data ./data.yaml --epochs 100 --imgsz 640 --device 0 1
```

─Ensure your data.yaml file is correctly set with dataset paths and classes.

─You can adjust training parameters like epochs, imgsz (image size), and device as needed.



##Detect with Webcam

```bash
python src/detect_webcam.py --weights models/best.pt
```

##Detect with Images

```bash
python src/detect_image.py --weights models/best.pt --source path/to/images
```



#Running Fire Detection


On Images
Run inference on individual images using the detection script:

```bash
python src/detect_image.py --source path/to/image.jpg --weights models/best.pt --conf 0.5
```




On Webcam (Real-time Detection)
To run detection on webcam feed:


```bash
python src/detect_webcam.py --weights models/best.pt --conf 0.5
```




#Notes


Replace "yolov9t.pt" with your chosen pretrained model or custom trained weights (best.pt).

The data.yaml file should point to your dataset paths and class names.

Adjust confidence threshold (--conf) as needed.

For Raspberry Pi deployment, use YOLO v9- tiny weights and the tiny model for better performance.




#Dataset

The dataset is annotated using Roboflow in YOLO format with images and labels structured appropriately.





#Results

Check the results/ folder for evaluation metrics like confusion matrix, precision-recall curves, and sample prediction images.




#License

This project uses the MIT License.




#Contact

For questions or contributions, please open an issue or contact hoodilyas1@gmail.com.


