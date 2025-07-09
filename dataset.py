# Custom dataset class
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

DATASET_PATH = "./data/results/"

# Loads the data from the folder on the path
# Has default transform specifically for inference/testing
class TestDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=dir_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label = self.dataset.imgs[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            default_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
            image = default_transform(image)

        return image, label

# Expects dataset (or subset) already on the input
class CustomImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        return img, label

    def __len__(self):
        return len(self.dataset)