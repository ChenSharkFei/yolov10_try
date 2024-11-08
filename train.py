from ultralytics import YOLO

def main():
    try:
        # Load a model
        model = YOLO("yolov10n.pt")

        # Train the model
        train_results = model.train(
            data=r'E:\yolov10\yolov10-main\data\data.yaml',  # path to dataset YAML
            epochs=5,  # number of training epochs
            batch=4,
            imgsz=640,  # training image size
            device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
            close_mosaic=0,
        )

        # Print training results
        print("Training Results:", train_results)

        # Evaluate model performance on the validation set
        metrics = model.val()

        # Print evaluation metrics
        print("Evaluation Metrics:", metrics)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()