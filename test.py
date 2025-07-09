# Final testing of the model

import torch
from torch.utils.data import DataLoader, Subset

import training
from training import MODEL_PATH
from dataset import DATASET_PATH, CustomImageDataset
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))
    model = torch.load(MODEL_PATH, weights_only=False)
    model.to(device)
    model.eval()

    # No additional transforms are used on the test set
    basic_transform = A.Compose([
        A.Resize(64, 64),
        A.ToFloat(),
        ToTensorV2(),
    ])

    # Prepare the dataset
    whole_dataset = datasets.ImageFolder(root=DATASET_PATH)
    test_indices = torch.load('test_indices.pth')
    test_subset = Subset(whole_dataset, test_indices)
    test_dataset = CustomImageDataset(test_subset, transform=basic_transform)
    test_loader = DataLoader(test_dataset)

    # Get the accuracies (also per-class)
    a = training.accuracy(model, test_loader, device, True)

    overall_acc = a['total']
    per_class_acc = a['per_class']
    print('Accuracy on test images: ', str(overall_acc))
    for c, acc in per_class_acc.items():
        print('Accuracy on class ', c,  str(acc))

