# Chest X-Ray Image Classification

A TensorFlow/Keras project that classifies chest X-ray images into three categories: **COVID**, **Normal**, and **Viral Pneumonia** (`virus`). The notebook compares a custom convolutional neural network (CNN) with transfer learning using a pretrained DenseNet201 model.

## Project overview

The notebook performs the following steps:

1. Loads images from Google Drive using `image_dataset_from_directory`.
2. Creates an 90% training / 10% validation split.
3. Resizes images to 256 × 256 pixels and uses batches of 32.
4. Displays sample images and applies data augmentation (random horizontal flip, rotation, and zoom).
5. Trains and evaluates two image-classification models:
   - **Custom CNN:** convolution, max-pooling, dropout, and dense layers.
   - **DenseNet201:** ImageNet-pretrained DenseNet201 used as a frozen feature extractor, followed by global average pooling and a classification head.

## Dataset

The notebook expects the dataset to be stored in Google Drive at:

```text
MyDrive/
└── train/
    ├── covid/
    │   ├── image_001.jpg
    │   └── ...
    ├── normal/
    │   ├── image_001.jpg
    │   └── ...
    └── virus/
        ├── image_001.jpg
        └── ...
```

The folder names are used as class labels. The notebook output shows **1,199 images across 3 classes**:

- `covid`
- `normal`
- `virus`

TensorFlow reported 1,080 training images and 119 validation images using a 10% validation split and random seed `100`.

> **Dataset note:** This repository does not include the image dataset. Make sure you have permission to use and share the dataset, and do not upload private or identifiable medical information.

## Models and configuration

| Setting | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Image size for initial dataset pipeline | 256 × 256 |
| DenseNet201 input size | 224 × 224 |
| Batch size | 32 |
| Validation split | 10% |
| Random seed | 100 |
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Maximum epochs | 100 |
| Early stopping | Monitors training loss; patience = 3 |

### Custom CNN

The first model uses data augmentation, pixel rescaling, three convolutional blocks, max pooling, dropout, a 128-unit dense layer, and a three-unit softmax output layer.

### DenseNet201 transfer learning

The second model uses DenseNet201 with ImageNet weights and `include_top=False`. The pretrained base is frozen; a classification head with global average pooling, a 128-unit dense layer, dropout, and a three-class softmax layer is trained on the X-ray dataset.

## Results

The notebook's saved outputs report the following evaluation results:

| Model | Evaluation loss | Evaluation accuracy |
|---|---:|---:|
| Custom CNN | 0.2489 | 91.60% |
| DenseNet201 | 0.2200 | 91.60% |

These values come from `model.evaluate(validation_set)` in the notebook. **They are validation-set results, not results from an independent test set.** They should not be described as test accuracy.

The notebook does not save model files or report additional metrics such as precision, recall, F1-score, a confusion matrix, or per-class performance.

## Run the notebook

The notebook was written for **Google Colab** and uses Google Drive for dataset access.

1. Upload `Xray.ipynb` to Google Colab.
2. Put the dataset in Google Drive using the folder structure shown above.
3. Open the notebook and run the cells from top to bottom.
4. When prompted, authorize Google Drive access.
5. If your dataset is stored somewhere else, update `base_dir` in the notebook.
6. Run the training and evaluation cells.

The first model trains on 256 × 256 images. The DenseNet201 model resizes the images to 224 × 224 inside the model.

## Run locally (optional)

For local use, install the packages in `requirements.txt`, then adapt the notebook's Google Drive paths and Drive-mount cell to point to a local dataset directory. The `google.colab` module is specific to Colab and is not needed as a pip dependency for a local setup.

## Important limitations

- This is an educational image-classification project, **not a medical diagnostic system**.
- The reported evaluation uses the validation split; an independent held-out test set is needed for a stronger estimate of generalization.
- Accuracy alone does not show how well the model identifies each class. Consider adding a confusion matrix, precision, recall, F1-score, and class-wise evaluation.
- The notebook does not include a separate inference script, saved model checkpoint, or deployment interface.

## Repository contents

```text
.
├── Xray.ipynb
├── README.md
└── requirements.txt
```

## Disclaimer

This project is for learning and experimentation only. It must not be used to diagnose, screen, or make treatment decisions for patients. Any real-world medical application would require appropriate clinical validation, expert oversight, privacy safeguards, and regulatory review.
