# Expiration Date Recognition

A deep learning-based solution for detecting and recognizing expiration dates, production dates, and other date-related information on product packaging using Faster R-CNN.

## Features

- Detection of multiple date-related fields:
  - Expiration dates
  - Due dates
  - Production codes
  - Product information
- Real-time visualization of detections
- Support for various image formats
- Pre-trained model available

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Windows 10 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/exp-date-recognition.git
cd exp-date-recognition
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Project Structure
```
exp-date-recognition/
├── src/
│   └── exp_date_recognition/
│       ├── __init__.py
│       ├── @detect.py        # Detection and visualization functions
│       ├── dataset.py        # Dataset handling
│       ├── train.py          # Training pipeline
│       └── transformations.py # Data augmentation
├── checkpoints/              # Model checkpoints
├── requirements.txt          # Project dependencies
├── download_checkpoint.py    # Script to download pre-trained model
└── test_model.py            # Testing script
```

## Quick Start

1. First, download the pre-trained model:
```python
# download_checkpoint.py
import gdown

url = 'YOUR_GDRIVE_LINK'
output = 'checkpoints/model.pth'
gdown.download(url, output, quiet=False)
```

2. Test the model on a single image:
```python
# test_model.py
from exp_date_recognition.detect import load_model, visualize_detections

# Load the pre-trained model
model, device = load_model('checkpoints/model.pth')

# Test on an image
image_path = 'path/to/your/image.jpg'
boxes, scores, labels = visualize_detections(image_path, model)
```

## Training Your Own Model

1. Prepare your dataset in the following structure:
```
dataset/
├── train/
│   ├── images/          # Training images
│   └── annotations.json # Training annotations
└── evaluation/
    ├── images/          # Validation images
    └── annotations.json # Validation annotations
```

2. Format your annotations.json files as follows:
```json
{
    "image1.jpg": {
        "ann": [
            {
                "bbox": [x_min, y_min, x_max, y_max],
                "cls": "date"
            }
        ]
    }
}
```

3. Train the model:
```python
from exp_date_recognition.train import train_model

# Start training
model = train_model(
    data_dir='path/to/dataset',
    num_epochs=10,
    batch_size=8
)
```

## Requirements

Create a requirements.txt file with the following dependencies:
```txt
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
numpy>=1.19.0
matplotlib>=3.3.0
albumentations>=1.0.0
gdown>=4.7.1
```

## API Reference

### Detection Module

```python
from exp_date_recognition.detect import load_model, detect_dates, visualize_detections

# Load model
model, device = load_model(model_path='checkpoints/model.pth', num_classes=5)

# Detect dates
boxes, scores, labels, image = detect_dates(
    model, 
    image_path='path/to/image.jpg',
    confidence_threshold=0.5
)

# Visualize results
boxes, scores, labels = visualize_detections(
    image_path='path/to/image.jpg',
    model=model,
    idx_to_class={1: 'date', 2: 'due', 3: 'code', 4: 'prod'}
)
```

## Troubleshooting

1. CUDA Out of Memory
```python
# Reduce batch size in train.py
train_model(data_dir, batch_size=4)  # Default is 8
```

2. Import Errors
- Ensure you're in the project root directory
- Verify virtual environment is activated:
```bash
.\venv\Scripts\activate
```

3. GPU Issues
- Check CUDA installation:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Common Issues and Solutions

1. ModuleNotFoundError:
```bash
pip install -e .
```

2. CUDA version mismatch:
- Check your CUDA version:
```bash
nvidia-smi
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- PyTorch team for the Faster R-CNN implementation
- Torchvision for the pre-trained models
- Albumentations for image augmentation

## Contact

Your Name - your.email@example.com
Project Link: [https://github.com/yourusername/exp-date-recognition](https://github.com/yourusername/exp-date-recognition)