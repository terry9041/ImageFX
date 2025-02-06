# ImageFX

## Overview
ImageFX is a Python-based Lightroom clone for opening and processing BMP images. The application provides various image processing operations, including grayscale conversion, dithering, edge detection, and more. It also includes an interactive UI with sliders for fine-tuning certain effects.

## Features
- **Basic Operations:**
  - Open BMP images
  - Display original image
  - Save processed images
- **Core Processing Functions:**
  - Grayscale conversion
  - Ordered dithering
  - Auto-level adjustment
- **Optional Image Effects:**
  - Invert Colors
  - Sepia Tone
  - Fish Eye Effect
  - Mosaic Effect
  - Sobel Edge Detection
  - Canny Edge Detection
  - Blur Image
  - Histogram Equalization
  - Unsharp Masking
  - Glitch Effect
- **Interactive Sliders:**
  - Adjust parameters for Unsharp Masking
  - Real-time preview of adjustments

## Installation
### Requirements
Ensure you have Python installed along with the following dependencies:
```sh
pip install numpy opencv-python tkinter
```

### Running the Application
```sh
python bmp_viewer.py
```

## Usage
1. Launch the application.
2. Open a BMP image using the file menu.
3. Apply various image processing effects from the menu options.
4. Use sliders to fine-tune effects like Unsharp Masking.
5. Save the processed image if needed.

## Controls
- **Menu Options:** Access various image effects and enhancements.
- **Sliders:** Adjust parameters like amount, radius, and threshold for Unsharp Masking.
- **Apply Button:** Apply real-time changes from the sliders.

## Contributing
Feel free to fork and contribute improvements. Submit a pull request with detailed explanations of your changes.

## License
This project is licensed under the MIT License.

