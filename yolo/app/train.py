from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11s.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="/ultralytics/data/coco.yaml",  # Path to dataset configuration file
    epochs=300,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
# metrics = model.val()

# Perform object detection on an image
# results = model("/ultralytics/data/images/val/2c7516e0d1a68c070a26b63ce6c313541abc62f79851eac3586c68622d39db26_3025.jpg")  # Predict on an image
# results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model