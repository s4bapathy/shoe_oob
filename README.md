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
