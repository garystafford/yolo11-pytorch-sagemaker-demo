# Real-Time Object Detection with YOLO11 and Amazon SageMaker

Learn how to deploy and performance test the state-of-the-art YOLO11 object detection model to Amazon SageMaker AI using PyTorch for production-ready, real-time inference with GPU acceleration. For more information see the blog post: [Real-Time Object Detection with YOLO11 and Amazon SageMaker AI](https://garystafford.medium.com/real-time-object-detection-with-yolo11-and-amazon-sagemaker-38b7476c1e2f).

![Sample Output](./previews/sample_images_01_detected.jpg)

![Sample Output](./previews/sample_images_04_detected.jpg)

## Features

- Downloads pre-trained YOLO11l model weights
- Creates SageMaker-compatible model artifact
- Deploys real-time endpoint with GPU acceleration (ml.g4dn.xlarge)
- Custom PyTorch inference handler (framework version 2.6.0)
- Supports JPEG and PNG image formats
- Includes performance testing with Locust

## Contents

- `deploy_yolo.ipynb` - End-to-end deployment notebook
- `code/` - SageMaker model code directory
  - `inference.py` - Custom SageMaker inference handler
  - `requirements.txt` - Python dependencies
- `sample_images/` - Sample images for testing
- `locust_scripts/` - Performance testing scripts
- `previews/` - Sample detection results

## Inference Pipeline

```text
┌─────────────────┐
│   Image Input   │ (JPEG/PNG)
│   (HTTP POST)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   input_fn()    │ Decode & validate image
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLO11l Model  │ Run inference on GPU
│    (PyTorch)    │ with confidence threshold
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  output_fn()    │ Format detection results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JSON Response  │ Bounding boxes, classes,
│   (HTTP 200)    │ confidence scores
└─────────────────┘
```

## Prerequisites

- AWS account with SageMaker access
- Python 3.12+
- Jupyter notebook environment

## Usage

Run the Jupyter notebook `deploy_yolo.ipynb` to:

1. Download YOLO11l model weights
2. Package model artifact with inference code
3. Deploy to SageMaker real-time endpoint
4. Test object detection on sample images

Optionally, use `locust_scripts/` for performance and load testing of the deployed endpoint.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

The contents of this repository represent my viewpoints and not those of my past or current employers, including Amazon Web Services (AWS). All third-party libraries, modules, plugins, and SDKs are the property of their respective owners.
