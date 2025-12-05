# Real-Time Object Detection with YOLO11 and Amazon SageMaker

Learn how to deploy and performance test the state-of-the-art YOLO11 object detection model to Amazon SageMaker AI using PyTorch for production-ready, real-time inference with GPU acceleration.

![Sample Output](./previews/sample_images_01_detected.jpg)

![Sample Output](./previews/sample_images_04_detected.jpg)

## Contents

- `deploy_yolo.ipynb` - End-to-end deployment notebook
- `inference.py` - Custom SageMaker inference handler
- `requirements.txt` - Python dependencies
- `sample_images/` - Sample images for testing

## Features

- Downloads pre-trained YOLO11 weights
- Creates SageMaker-compatible model artifact
- Deploys real-time endpoint with GPU acceleration
- Custom inference code
- Supports JPEG and PNG image formats

## Usage

Run the Jupyter notebook `deploy_yolo.ipynb` to:

1. Download YOLO11 model weights
2. Package model artifact with inference code
3. Deploy to SageMaker endpoint
4. Test object detection on sample images

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

The contents of this repository represent my viewpoints and not those of my past or current employers, including Amazon Web Services (AWS). All third-party libraries, modules, plugins, and SDKs are the property of their respective owners.
