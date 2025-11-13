
# Chest X-ray Pneumonia Detection (Sample Project)

**What this repo contains**

- `sample/` - placeholder for a few sample chest x-ray images (add your own small images here)
- `source/` - Python package with dataset utilities and model/training code
- `tests/` - simple unit tests for core functions
- `.gitignore`, `Makefile`, `requirements.txt` - helpers for development
- `main.py` - entrypoint script to train/evaluate the model

---

## Project overview

This project trains a simple convolutional neural network to classify chest X-ray images
as **NORMAL** or **PNEUMONIA**. The code is adapted from a training notebook and designed
to be runnable from the command line or as importable modules for further development.

Key features:
- `ImageDataGenerator` based pipeline with augmentation
- Balanced training via `class_weight`
- Train/validation split and test evaluation
- Model saving and simple evaluation metrics (accuracy, precision, recall, f1)

---

## Quickstart

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Place your dataset in `data/chest_xray` with subfolders `train/`, `val/`, and `test/`
   each containing `NORMAL/` and `PNEUMONIA/` directories of images (same layout as used commonly).

3. Run training (example):

```bash
python main.py --mode train --data_dir data/chest_xray --epochs 10 --batch_size 32
```

4. To evaluate on the test set:

```bash
python main.py --mode evaluate --data_dir data/chest_xray
```

---

## Files of interest

- `source/data_loader.py` - dataset and generator creation
- `source/model.py` - model architecture and compile utilities
- `source/train.py` - training loop and history saving
- `main.py` - CLI wrapper to train/evaluate

---

## Notes & suggestions

- If you have limited GPU memory, reduce `TARGET_SIZE` or `batch_size`.
- Consider using a pretrained backbone (e.g., MobileNetV2) for higher accuracy and faster convergence.
- The included tests are simple smoke tests; expand them before deploying.

