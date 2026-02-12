# Face_alignment

# Face Alignment Project

University project for facial landmark detection using a classical machine-learning pipeline in Python.

## Overview

This project trains a landmark predictor on face images, visualizes predicted points on sample images, applies simple eye/lip color overlays, and exports predictions for the test set to CSV.

Main workflow implemented in `Face_Alignment_(1).ipynb`:

1. Download and validate datasets (`training_images.npz`, `test_images.npz`, `examples.npz`).
2. Train a per-landmark linear regression model using SIFT descriptors.
3. Evaluate predictions on a small validation split with mean Euclidean distance.
4. Visualize predicted landmarks on example images.
5. Apply color fill to eyes/lips using predicted landmarks.
6. Predict landmarks for the test set and save `results.csv`.

## Project Structure

- `Face_Alignment_(1).ipynb`: Main notebook with full training/evaluation/inference pipeline.
- `results.csv`: Generated output file with predicted test landmarks (created after running inference).
- `train_model.mdl`: Serialized trained model (created after training).

## Requirements

- Python 3.9+ (recommended)
- Jupyter Notebook or Google Colab
- Python packages:
  - `numpy`
  - `opencv-contrib-python`
  - `matplotlib`
  - `tqdm`
  - `scikit-learn`

Install locally:

```bash
pip install numpy opencv-contrib-python matplotlib tqdm scikit-learn
```

## How to Run

1. Open `Face_Alignment_(1).ipynb` in Jupyter or Colab.
2. Run cells in order from top to bottom.
3. The notebook will:
   - download data files via `wget`
   - verify file checksums
   - train and save a model (`train_model.mdl`)
   - show landmark and color-overlay visualizations
   - generate `results.csv` for test images

## Output Format

`results.csv` contains one row per test image and flattened landmark coordinates:

- shape: `(554, 88)`
- interpretation: `44` points per image, each as `(x, y)`

## Notes

- SIFT features are used at mean landmark positions estimated from training labels.
- Runtime depends on hardware; training can take longer on CPU-only environments.
- If dataset URLs become unavailable, replace download links with your instructor-provided mirrors.
