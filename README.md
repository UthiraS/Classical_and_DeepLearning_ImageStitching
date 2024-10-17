# RBE/CS549 Computer Vision - AutoPano Project

This project presents two different approaches to creating seamless panorama images by stitching two or more images together.

## Phase 1: Classical Method

This phase implements the traditional approach to find a homography matrix between a set of two images.

### Steps to run:

```bash
cd Phase1/Code
python Wrapper.py
```

## Phase 2: Deep Learning-based Homography Estimation

This phase describes the implementation of a supervised and an unsupervised deep learning approach for estimating homography between synthetically generated data.

### Steps to run:
For Supervised Model
```bash
cd Phase2/Code
python Wrapper.py --NumFeatures 128 --Data Train --ModelType 'Sup'
```
For UnSupervised Model
```bash
cd Phase2/Code
python Wrapper.py --NumFeatures 128 --Data Train --ModelType 'UnSup'
```
## Project Overview

- **Objective**: Implement classical and deep learning techniques for image stitching and homography estimation.
- **Technologies Used**: Python, NumPy, SciPy, OpenCV, PyTorch, CNN
- **Key Features**:
  - Corner detection
  - RANSAC (Random Sample Consensus)
  - ANMS (Adaptive Non-Maximal Suppression)
  - Feature detection (ANMS, Feature Descriptor)
  - Feature matching
  - Homography estimation

## Achievements

- Improved image stitching quality using advanced techniques.
- Successfully implemented both classical and deep learning approaches.

## Additional Notes

- The deep learning approach uses synthetically generated data for training.
- Both supervised and unsupervised methods are explored in the deep learning phase.

For more details on the implementation and theory, please refer to the project report.



