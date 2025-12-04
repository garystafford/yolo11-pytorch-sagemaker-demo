import os
import json
import time
import logging
from io import BytesIO
from typing import Any, List

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONF_THRESHOLD = 0.25
SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/jpg")
MAX_IMAGE_DIMENSION = 10000  # Prevent extremely large images


def model_fn(model_dir: str) -> YOLO:
    """Load and prepare YOLO model for inference.

    Args:
        model_dir: Directory containing the model weights

    Returns:
        Configured YOLO model ready for inference
    """
    logger.info("Loading YOLO model from %s", model_dir)

    weights_name = os.getenv("YOLO_MODEL", "yolo11l.pt")
    weights_path = os.path.join(model_dir, weights_name)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    try:
        # Load YOLO11 model
        model = YOLO(weights_path)
    except Exception as e:
        logger.error("Failed to load YOLO model: %s", e)
        raise RuntimeError(f"Model loading failed: {e}") from e

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    model.to(device)

    # Try to fuse layers for inference speedup (may not work for all models)
    try:
        model.fuse()
        logger.info("Model layers fused successfully")
    except Exception as e:
        logger.warning("Could not fuse model layers: %s", e)

    model.eval()

    # Cache class names for use in output_fn
    model.class_names = model.names

    # Read default conf from env, fallback to constant
    try:
        model.conf_thres = float(os.getenv("YOLO_CONF", str(DEFAULT_CONF_THRESHOLD)))
    except ValueError:
        logger.warning(
            "Invalid YOLO_CONF value, using default: %s", DEFAULT_CONF_THRESHOLD
        )
        model.conf_thres = DEFAULT_CONF_THRESHOLD

    logger.info(
        "Model loaded successfully with confidence threshold: %.2f", model.conf_thres
    )

    return model


def input_fn(request_body: bytes, request_content_type: str) -> np.ndarray:
    """Decode image from request body.

    Args:
        request_body: Raw bytes of the image
        request_content_type: MIME type of the image

    Returns:
        Numpy array in BGR format ready for YOLO inference
    """
    if request_content_type not in SUPPORTED_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type: {request_content_type}. "
            f"Supported types: {SUPPORTED_CONTENT_TYPES}"
        )

    # Decode image bytes using Pillow
    try:
        img = Image.open(BytesIO(request_body))

        # Validate image dimensions
        width, height = img.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValueError(
                f"Image dimensions ({width}x{height}) exceed maximum allowed "
                f"({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            )

        logger.info("Decoded image: %dx%d, mode=%s", width, height, img.mode)

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            logger.debug("Converting image from %s to RGB", img.mode)
            img = img.convert("RGB")

        # Convert PIL Image to numpy array for YOLO model (expects BGR format)
        img_array = np.array(img)
        # Convert RGB to BGR for compatibility with YOLO's expected format
        img_array = img_array[:, :, ::-1]

        return img_array

    except Image.UnidentifiedImageError as e:
        raise ValueError(f"Invalid image format or corrupted image data: {e}")
    except Exception as e:
        logger.error("Image decoding failed: %s", e)
        raise ValueError(f"Failed to decode image: {e}")


def predict_fn(input_data: np.ndarray, model: YOLO) -> List[Any]:
    """Run inference on input image.

    Args:
        input_data: Image as numpy array in BGR format
        model: Loaded YOLO model

    Returns:
        List of Ultralytics Results objects
    """
    conf_threshold = getattr(model, "conf_thres", DEFAULT_CONF_THRESHOLD)
    logger.info(
        "Running inference on image shape=%s, conf_threshold=%.2f",
        input_data.shape,
        conf_threshold,
    )

    start = time.perf_counter()
    try:
        with torch.no_grad():
            results = model(input_data, conf=conf_threshold)
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise RuntimeError(f"Model inference failed: {e}") from e

    elapsed = (time.perf_counter() - start) * 1000

    # Count detections for logging
    total_detections = sum(
        len(r.boxes) if hasattr(r, "boxes") and r.boxes is not None else 0
        for r in results
    )
    logger.info(
        "Inference completed in %.2f ms, detected %d objects", elapsed, total_detections
    )

    return results


def output_fn(prediction_output: List[Any], content_type: str) -> str:
    """Format prediction results as JSON.

    Args:
        prediction_output: List of Ultralytics Results objects
        content_type: Desired output content type

    Returns:
        JSON string with detections and metadata
    """
    detections = []
    inference_time_ms = 0
    image_shape = None

    # Prediction_output is a list of Ultralytics Results objects
    for result in prediction_output:
        # Extract metadata if available
        if hasattr(result, "speed") and result.speed:
            inference_time_ms = result.speed.get("inference", 0)
        if hasattr(result, "orig_shape"):
            image_shape = result.orig_shape

        # Get class names with simplified lookup
        names = _get_class_names(result)
        names_is_dict = isinstance(names, dict)
        names_is_seq = isinstance(names, (list, tuple))

        # Process boxes if available
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes_data = result.boxes.data
            if boxes_data is not None and len(boxes_data) > 0:
                # Convert to numpy once (more efficient than per-item conversion)
                boxes_np = boxes_data.cpu().numpy()

                for box_data in boxes_np:
                    x1, y1, x2, y2, conf, cls_id = box_data[:6]
                    cls_id = int(cls_id)

                    # Map class id -> label string
                    label = _get_class_label(cls_id, names, names_is_dict, names_is_seq)

                    detections.append(
                        {
                            "box": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(conf),
                            "class_id": cls_id,
                            "label": label,
                        }
                    )

    # Build response with metadata
    response = {
        "detections": detections,
        "metadata": {
            "count": len(detections),
            "inference_time_ms": inference_time_ms,
        },
    }

    if image_shape is not None:
        response["metadata"]["image_shape"] = {
            "height": int(image_shape[0]),
            "width": int(image_shape[1]),
        }

    logger.info("Returning %d detections", len(detections))

    return json.dumps(response)


def _get_class_names(result: Any) -> Any:
    """Extract class names from result object."""
    return (
        getattr(result, "names", None)
        or getattr(getattr(result, "model", None), "names", None)
        or getattr(getattr(result, "model", None), "class_names", None)
    )


def _get_class_label(
    cls_id: int, names: Any, names_is_dict: bool, names_is_seq: bool
) -> str:
    """Get class label string from class ID."""
    if names_is_dict:
        return names.get(cls_id, str(cls_id))
    elif names_is_seq and 0 <= cls_id < len(names):
        return names[cls_id]
    else:
        return str(cls_id)
