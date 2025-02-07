# Expiration Date Recognition

A computer vision project that detects and recognizes expiration dates on product packaging using Faster R-CNN and OCR.

## Setup Guide for Windows . 

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
   git clone https://github.com/ankush2992/exp-date-recognition.git
   cd exp-date-recognition
   ```

2. **Proceed to 3**
   
3. **Install Required Packages**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Download Model Checkpoint**
   ```cmd
   pip install gdown
   python download_checkpoint.py
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

Loading model...
Model loaded successfully
Processing image...
Found date (confidence: 0.92): 15-09-2021


## Troubleshooting Common Windows Issues

### 1. Tesseract Not Found Error
Error: tesseract is not installed or it's not in your PATH

**Solution:**
- Verify Tesseract installation path
- Add to PATH: `C:\Program Files\Tesseract-OCR`
- Restart code editor

## Project Structure

exp-date-recognition/

├── checkpoints/              # Downloaded model files

├── images for testing/       # some sample images inside - copy one of them in test_image folder and rename it to test_image.jpg   👈👈👈 

├── test_images/             # Your test images go here

├── src/

├── requirements.txt

├── download_checkpoints.py

└── test_model.py



## Additional Notes
- 👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇
- Pick any images and rename it to test_image.jpg and copy that image inside test_images [Folder]  , and run the code . MAKE SURE - ONLY 1 IMAGE AT A TIME - named test_image.jpg indise test_images [FOLDER] will be taken for detection...🤦🏽‍♂️🤦🏽‍♂️
- 👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆

- The model checkpoint is ~300MB and will be downloaded during setup
- First-time run might be slower due to CUDA initialization
- For best results, ensure images are well-lit and dates are clearly visible

## Support

If you encounter any issues:
1. Ensure all installation steps were followed
2. contact me 


## License
This project is licensed under the MIT License.
