# Shoe Detection with GroundingDINO

This project demonstrates how to use the `GroundingDINO` model from the `autodistill_grounding_dino` package to detect shoes in images. The script processes an input image, detects shoes, annotates the image with bounding boxes and confidence scores, and extracts detected shoe regions into individual images.

## Installation

Ensure you have the required packages installed:

```bash
pip install autodistill_grounding_dino autodistill.detection opencv-python-headless supervision
```

## Usage

### Basic Usage

To run the script with basic settings, use the following command:

```bash
python shoe_detection.py
```

This will process the image `test.jpg`, annotate it with detected shoe regions, and save the annotated image as `annotated_test.jpg`. Detected shoe regions with a confidence score of 0.5 or higher will be extracted and saved in the `extracted_shoes` directory.

### Advanced Usage

You can customize the detection process by specifying optional thresholds and prompts:

```python
# Example usage with custom thresholds
predictions, _ = detect_shoes(
    image_path="test.jpg", 
    prompt="shoes, sneakers, boots, footwear",
    box_threshold=0.3,
    text_threshold=0.25,
    confidence_threshold=0.6,  # Higher threshold for extraction
    output_path="custom_threshold_output.jpg",
    extract_dir="high_confidence_shoes"
)
```

### Function Parameters

- `image_path` (str): Path to the input image.
- `prompt` (str): Text prompt for detecting shoes. Default is "shoe, footwear".
- `output_path` (str): Path to save the annotated output image.
- `box_threshold` (float, optional): Threshold for bounding box detection.
- `text_threshold` (float, optional): Threshold for text detection.
- `confidence_threshold` (float): Minimum confidence score for extracting detected regions. Default is 0.0.
- `extract_dir` (str): Directory to save extracted shoe images. Default is "extracted_shoes".

## Example

Here is an example of how to use the `detect_shoes` function in a script:

```python
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import cv2
import supervision as sv
import os
import time
import numpy as np

def detect_shoes(
    image_path, 
    prompt="shoe, footwear",
    output_path="annotated_output.jpg",
    box_threshold=None,  # Optional threshold parameters
    text_threshold=None,
    confidence_threshold=0.0,  # Minimum confidence to extract
    extract_dir="extracted_shoes"  # Directory to save extracted images
):
    print(f"Processing image: {image_path}")
    start_time = time.time()
    
    # Initialize model with optional thresholds
    model_params = {"ontology": CaptionOntology({"shoe": prompt})}
    if box_threshold is not None:
        model_params["box_threshold"] = box_threshold
    if text_threshold is not None:
        model_params["text_threshold"] = text_threshold
    
    # Create model with appropriate parameters
    model = GroundingDINO(**model_params)
    
    # Make predictions
    predictions = model.predict(image_path)
    print(f"Found {len(predictions)} detections")
    
    # Load image and create annotator
    image = cv2.imread(image_path)
    
    # Create annotator with confidence display
    box_annotator = sv.BoxAnnotator()
    
    # Create label annotator for adding confidence scores
    label_annotator = sv.LabelAnnotator()
    
    # Create labels with confidence scores
    labels = []
    if hasattr(predictions, 'confidence') and len(predictions) > 0:
        labels = [f"shoe: {conf:.2f}" for conf in predictions.confidence]
    else:
        labels = [f"shoe" for _ in range(len(predictions))]
    
    # Annotate the image with boxes
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=predictions)
    
    # Add confidence labels
    annotated_image = label_annotator.annotate(
        scene=annotated_image, 
        detections=predictions, 
        labels=labels
    )
    
    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)
    
    # Create directory for extracted images if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract and save individual shoe images based on confidence threshold
    extracted_count = 0
    if len(predictions) > 0:
        for i, bbox in enumerate(predictions.xyxy):
            # Skip if confidence is below threshold
            if hasattr(predictions, 'confidence') and predictions.confidence[i] < confidence_threshold:
                continue
                
            try:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Ensure coordinates are within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                # Extract the region
                crop = image[y1:y2, x1:x2]
                
                # Generate filename with confidence if available
                if hasattr(predictions, 'confidence'):
                    conf = predictions.confidence[i]
                    crop_path = os.path.join(extract_dir, f"shoe_{i}_conf{conf:.2f}.jpg")
                else:
                    crop_path = os.path.join(extract_dir, f"shoe_{i}.jpg")
                
                # Save the cropped image
                cv2.imwrite(crop_path, crop)
                extracted_count += 1
                
            except Exception as e:
                print(f"Error extracting detection {i}: {e}")
    
    end_time = time.time()
    print(f"Detection completed in {end_time - start_time:.2f} seconds")
    print(f"Annotated image saved to {output_path}")
    print(f"Extracted {extracted_count} images with confidence >= {confidence_threshold} to {extract_dir}/")
    
    return predictions, annotated_image

# Example usage
if __name__ == "__main__":
    IMAGE_NAME = "test.jpg"
    OUTPUT_IMAGE_NAME = "annotated_test.jpg"
    
    # Basic usage with confidence display and extraction
    predictions, _ = detect_shoes(
        image_path=IMAGE_NAME,
        output_path=OUTPUT_IMAGE_NAME,
        confidence_threshold=0.5,  # Only extract detections with confidence ≥ 0.5
        extract_dir="extracted_shoes"
    )
    
    # Advanced usage with custom thresholds
    # predictions, _ = detect_shoes(
    #     image_path=IMAGE_NAME, 
    #     prompt="shoes, sneakers, boots, footwear",
    #     box_threshold=0.3,
    #     text_threshold=0.25,
    #     confidence_threshold=0.6,  # Higher threshold for extraction
    #     output_path="custom_threshold_output.jpg",
    #     extract_dir="high_confidence_shoes"
    # )
