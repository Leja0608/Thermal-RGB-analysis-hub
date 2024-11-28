# Thermal-RGB Analysis Hub

**Thermal-RGB Analysis Hub** is a project focused on the analysis and enhancement of thermal and RGB images using computer vision and image processing techniques. The main objective of this project is to facilitate the integration and registration of thermal images with RGB images to provide more accurate analysis in various applications such as thermal monitoring of systems and image quality enhancement.

## Project Contents

- **Thermal and RGB Image Analysis**: Using image processing techniques to align and improve thermal images.
- **Multimodal Registration**: Methods for aligning RGB images with thermal images to enhance data visualization.
- **Visualization**: Tools for visualizing thermal and RGB data within the same workspace.
- **Interactive Notebooks**: Jupyter notebooks for experimentation and execution of image analysis.

## Installation

### Prerequisites

To run this project, you need Python 3.9 and the following Python libraries. It is recommended to use a virtual environment to avoid dependency conflicts.

### Installation Steps:

1. **Clone the repository**:
   If you haven't cloned the repository yet, run the following command in your terminal:
   ```bash
   git clone https://github.com/Leja0608/Thermal-RGB-analysis-hub.git

2. **Install dependencies: Navigate to the project folder and run**:
    pip install -r requirements.txt

## Requirements

This project requires the following Python libraries:

- `numpy`
- `opencv-python`
- `matplotlib`
- `SimpleITK`

## Usage

1. Open a notebook:
   Navigate to the `notebooks/` directory and open the notebook of interest.

2. Run the code:
   Once the notebook is open, run the cells to perform the image analysis.

### Example:
```python
import cv2

rgb_image = cv2.imread('path_to_rgb_image.jpg')
thermal_image = cv2.imread('path_to_thermal_image.jpg')

# Image processing and alignment


## Contributing

Contributions are welcome. Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or fix (`git checkout -b feature/my-feature`).
3. Make changes and commit them (`git commit -am 'Add new feature'`).
4. Push the changes (`git push origin feature/my-feature`).
5. Create a pull request for review.
