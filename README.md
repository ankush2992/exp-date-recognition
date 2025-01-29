# Expiration Date Recognition

A computer vision project that detects and recognizes expiration dates on product packaging using Faster R-CNN and OCR.

## Setup Guide for Windows

### Prerequisites

1. **Python Installation**
   - Download and install Python 3.8 or higher from [Python.org](https://www.python.org/downloads/)
   - During installation, make sure to check ✅ "Add Python to PATH"

2. **Git Installation**
   - Download and install Git from [Git for Windows](https://gitforwindows.org/)

3. **Tesseract OCR Installation**
   - Download and install Tesseract from [Direct-Link](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)
   - Install it in a path without spaces (e.g., `C:\Program Files\Tesseract-OCR`)
   - Add Tesseract to System PATH:
     - Search for "Environment Variables" in Windows
     - Under System Variables, select "Path" and click "Edit"
     - Add new entry: `C:\Program Files\Tesseract-OCR`
     - Click OK to save

### Installation Steps

1. **Clone the Repository**
   ```cmd
   git clone https://github.com/your-username/exp-date-recognition.git
   cd exp-date-recognition
   ```

2. **Create Virtual Environment**
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Required Packages**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Download Model Checkpoint**
   ```cmd
   pip install gdown
   python download_checkpoints.py
   ```

### Testing the Installation

1. **Verify Tesseract Installation**
   ```cmd
   tesseract --version
   ```
   If this fails, check your PATH settings.

2. **Test the Model**
   ```cmd
   python test_model.py
   ```
   This will process sample images and show detected dates with confidence scores.

## Usage

### Running Date Detection

1. Place your test images in the `test_images` folder
2. Run:
   ```cmd
   python test_model.py
   ```
3. Results will show:
   - Detected date regions
   - Confidence scores
   - Extracted dates

### Expected Output
