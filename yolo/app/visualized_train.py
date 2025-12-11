from ultralytics import YOLO
import os

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Training parameters
total_epochs = 100
checkpoint_epochs = 10
data_yaml = "/ultralytics/data/coco.yaml"
val_img_path = "/ultralytics/data/images/val/2c7516e0d1a68c070a26b63ce6c313541abc62f79851eac3586c68622d39db26_3025.jpg"
device = "cpu"

# Create a directory to save sample predictions
os.makedirs("predicts", exist_ok=True)

# Start training in chunks
for current_epoch in range(0, total_epochs, checkpoint_epochs):
    print(f"\n=== Training from epoch {current_epoch} to {current_epoch + checkpoint_epochs} ===")
    
    # Resume is True after the first run if weights exist
    resume = current_epoch > 0 and os.path.exists("runs/detect/train/weights/last.pt")

    # Train for checkpoint_epochs
    model.train(
        data=data_yaml,
        epochs=current_epoch + checkpoint_epochs,
        imgsz=640,
        device=device,
        save=True,
        exist_ok=True,  # Allow overwriting runs
        resume=resume  # Resume training
    )

    # Reload best or last model to ensure we're using updated weights
    # Option 1: Use last checkpoint (most recent)
    model = YOLO("runs/detect/train/weights/last.pt")  # Or use 'best.pt' if you prefer best so far

    # Run prediction on a validation image
    print(f"Running inference after epoch {current_epoch + checkpoint_epochs}")
    results = model(val_img_path)

    # Save with unique name per epoch
    save_path = f"predicts/pred_epoch_{current_epoch + checkpoint_epochs}"
    results[0].save(save_path)  # Save image
    results[0].show()  # Optionally display

# Final evaluation
print("Final validation metrics:")
metrics = model.val()

# Export model
print("Exporting model...")
path = model.export(format="onnx")
print(f"Model exported to: {path}")
