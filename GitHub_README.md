# AI-Powered Cell Morphology Analyzer 🔬💻

## Overview
This project is a Computer Vision application designed to automate the morphological analysis of cells and embryos. By leveraging OpenCV and Python, the script processes microscopic images, detects cellular membranes, and classifies cells as "Healthy" or "Abnormal" based on mathematical circularity metrics.

This tool represents the powerful intersection of **Molecular Biology and Genetics** with **Computer Programming**—bridging the gap between laboratory wet-work (like analyzing cell lines) and automated bioinformatics pipelines.

## Features
* **Automated Edge Detection:** Uses Gaussian Blurring and Canny Edge Detection to isolate cellular structures from background noise.
* **Morphological Classification:** Calculates the circularity formula `(4 * π * Area) / (Perimeter^2)` to determine cell health.
* **Batch Object Counting:** Iterates through all detected contours, filtering out micro-noise, and provides a total count of healthy vs. abnormal structures.
* **Visual Overlay:** Outputs a visually mapped image with color-coded bounding contours (Green for healthy, Red for abnormal) and individual scores.

## Technologies Used
* Python 3
* OpenCV (`cv2`)
* Math module

## Example Output
The script takes a raw `.jpg` microscope image and outputs a processed image featuring:
* Total cell/organelle count.
* Color-coded segmentation.
* Individual morphological scores printed directly on the targets.

## Usage
1. Place your target image as `hucre.jpg` in the root directory.
2. Run the script: `python analiz.py`
3. Check the output file `gelismis_analiz.jpg` for the mapped results.

## Future Roadmap
* Integration with advanced deep learning models (CNNs) for anomaly detection.
* Expanded support for specific cell lines and automated cancer cell (e.g., U87) characterization.
