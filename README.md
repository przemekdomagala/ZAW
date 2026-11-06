# Computer Vision Laboratory

A collection of computer vision and image processing exercises and a final project, implemented in Python with OpenCV, NumPy, PyTorch, and TensorFlow/Keras. Topics range from classical feature detection and tracking to visual odometry, multi-modal detection, and deep learning-based approaches.

## Repository Structure

| Directory | Topic | Key Files |
|-----------|-------|-----------|
| `lab06-feature-detection-matching` | Harris corner detection and ORB-based feature matching | `feature_detection_matching.ipynb`, `plot_utils.py` |
| `lab07-dcf-tracking` | Discriminative Correlation Filter (DCF) object tracking, including training stage and spatial response maps | `dcf_tracking.ipynb`, `dcf_tracking.py`, `dcf_tracking_colab.ipynb` |
| `lab08-visual-odometry` | Monocular visual odometry: feature detection, matching, and essential matrix estimation | `visual_odometry.ipynb` |
| `lab09-generalized-hough-transform` | Generalized Hough Transform for shape (gear) detection using gradient orientation and amplitude | `generalized_hough_transform.ipynb` |
| `lab10-rgb-thermal-detection` | Object detection combining RGB and thermal imagery with multiple fusion strategies | `rgb_thermal_detection.ipynb`, `rgb_thermal_detection.py` |
| `lab11-hog-svm-detection` | Object detection/classification using HOG features, SVM, image pyramids, and non-maximum suppression | `hog_svm_detection.ipynb` |
| `lab12-siamese-tracking` | Deep learning-based object tracking with Siamese neural networks | `siamese_tracking.ipynb` |
| `lab13-event-cameras` | Processing and 3D visualization of event camera data streams | `event_camera_processing.ipynb` |
| `project-surface-defect-detection` | Final project: patch-based surface defect detection on leather using an autoencoder (MVTec AD-style evaluation) | `surface_defect_detection.ipynb` |

## Topics Covered

- **Feature detection and matching** - Harris corners, ORB descriptors
- **Classical tracking** - Discriminative Correlation Filters
- **Visual odometry** - camera pose estimation from image sequences
- **Shape detection** - Generalized Hough Transform
- **Multi-modal detection** - RGB-thermal image fusion
- **Machine learning detection** - HOG + SVM with sliding windows and pyramid search
- **Deep learning tracking** - Siamese network architecture
- **Neuromorphic vision** - event camera data processing
- **Anomaly detection** - autoencoder-based surface defect localization

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- SciPy
- scikit-learn
- PyTorch
- TensorFlow / Keras
- Matplotlib
- imutils

## Usage

Each directory is self-contained and can be run independently. Most modules are provided as Jupyter notebooks; a few include standalone Python scripts (see the table above).

```bash
# Install dependencies
pip install opencv-python numpy scipy scikit-learn torch tensorflow matplotlib imutils

# Launch a notebook
jupyter notebook lab06-feature-detection-matching/feature_detection_matching.ipynb
```

> **Note:** Some notebooks expect image datasets located in their working directory (e.g., `fontanna1.jpg`, `trybiki2.jpg`, `events.txt`, `leather/`). Place the required data files in the corresponding directory before running.
