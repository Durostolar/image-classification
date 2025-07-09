## Image Classification

#### A complete pipeline for an image classification task using CNNs: custom dataset preparation, preprocessing, dataset handling, model training and evaluation.

- **Framework**: PyTorch  
- **Model**: ResNet-18 (https://arxiv.org/abs/1512.03385) trained from scratch, no pre-trained weights used. Can be easily replaced with other architectures.  
 - **Dataset**: Built from the Cityscapes dataset (https://www.cityscapes-dataset.com/). Instances are not included here, for download visit the page. ***This project uses a custom dataset, not a benchmark.*** 
 - **Other libraries**: albumentations, torchvision, PIL, numpy

### Contents
-   `create_dataset.py`  
    Builds the image classification dataset from Cityscapes benchmark. Images and labels are generated from files originally created for class instance segmentation.
-   `dataset.py`  
    Custom dataset wrappers to simplify the application of transforms and inference.
-   `training.py`  
    Main training logic:
    -- Loads and splits the dataset (train/validation/test)
    -- Defines the model (ResNet-18), configures optimizer, scheduler, callbacks
    --   Trains the model and plots learning curves, saves the trained model
-   `test.py`  
    Loads the trained model and evaluates it on the test set, reporting overall and per-class accuracy.

### Results
Learning curve plot:
#
[![alt text]()]
Accuracy on test images:
| Class       | Accuracy (%)    |
|-------------|-----------------|
| **Overall** | **82.61**           |
| Class 0     | 78.43           |
| Class 1     | 78.95           |
| Class 2     | 86.67           |
| Class 3     | 63.16           |
| Class 4     | 94.74           |
| Class 5     | 89.32           |
| Class 6     | 64.71           |
| Class 7     | 85.71           |


