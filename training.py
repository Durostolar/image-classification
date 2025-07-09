import matplotlib.pyplot as plt

import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from dataset import CustomImageDataset
from dataset import DATASET_PATH
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_PATH = 'model.pt'

# Losses visualization
def plot_learning_curves(train_losses, validation_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.yscale('log')
    plt.legend()
    plt.savefig("learning_curves.png")

# Callback based on the validation loss
class BestmodelCallback:
    def __init__(self, model_path='best_model.pt'):
        self.best_valid_loss = float('inf')
        self.model_path = model_path
        
    def __call__(self, model, valid_loss):
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            torch.save(model, self.model_path)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, dev, save_path='model.pt'):
    train_losses = []
    validation_losses = []

    best_model = BestmodelCallback(save_path)
    
    # Similar to what was used in ResNet paper, divide lr by 10 when loss is stuck. 
    # Low patience id used as the dataset is fairly small
    scheduler = ReduceLROnPlateau(opt, patience=4)

    for epoch in range(epochs):
        # Training step
        loss = train(model, train_dl, loss_func, dev, opt)
        
        # Validation step
        val_loss = validate(model, valid_dl, loss_func, dev)

        train_losses.append(loss)
        validation_losses.append(val_loss)

        scheduler.step(val_loss)

        best_model(model, val_loss)

        print(f'epoch {epoch+1}/{epochs}, loss: {loss : .05f}, validation loss: {val_loss:.05f}')

        print('Accuracy on the train images: ', accuracy(model, train_dl,dev)['total'])
        print('Accuracy on the validation images: ', accuracy(model, valid_dl,dev)['total'])

    return train_losses, validation_losses

def accuracy(model, loader, dev, per_class=False):
    correct = 0 
    total = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if per_class:
                for label, pred in zip(labels, predicted):
                    label_idx = label.item()
                    class_correct[label_idx] = class_correct.get(label_idx, 0) + int(label == pred)
                    class_total[label_idx] = class_total.get(label_idx, 0) + 1

    overall_accuracy = 100 * correct/total

    per_class_accuracy = None
    if per_class:
        per_class_accuracy = {c: 100 * class_correct[c] / class_total[c]
                              for c in class_total}

    return {'total' : overall_accuracy, 'per_class' : per_class_accuracy}
    
def train(model, train_dl, loss_func, dev, opt):
        model.train()
        loss, size = 0, 0
        for b_idx, (xb, yb) in enumerate(train_dl):
            b_loss, b_size = loss_batch(model, loss_func, xb, yb, dev, opt)
            
            loss += b_loss * b_size
            size += b_size
            
        return loss / size
    
def validate(model, valid_dl, loss_func, dev):
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb, dev) for xb, yb in valid_dl]
            )
            
        return np.sum(np.multiply(losses, nums)) / np.sum(nums)

def loss_batch(model, loss_func, xb, yb, dev, opt=None):
    xb, yb = xb.to(dev), yb.to(dev)

    # Forward pass
    loss = loss_func(model(xb), yb)

    # Backward pass
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), len(xb)

# Prepares datasets splits and transforms
def setup_data_splits_and_transforms():
    # The dataset is created in two steps, so it can be easily split and each split can use different transforms if needed
    # First define general dataset for the whole input folder

    whole_dataset = datasets.ImageFolder(root=DATASET_PATH)

    # Split the data
    lengths = [0.8, 0.1, 0.1]
    torch.manual_seed(0)
    subsetA, subsetB, subsetC = random_split(whole_dataset, lengths)
    torch.save(subsetC.indices, 'test_indices.pth')

    # Transform with augumentations
    # The dataset is fairly small, without this, the model was quickly overfitting
    train_transform_augmentation = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.GaussNoise(),

        # Bigger size did not improve the performance
        A.Resize(64, 64),

        A.HorizontalFlip(),

        # Finalize
        A.ToFloat(),
        ToTensorV2(),
    ])

    train_dataset = CustomImageDataset(subsetA, transform=train_transform_augmentation)
    valid_dataset = CustomImageDataset(subsetB, transform=train_transform_augmentation)

    return train_dataset, valid_dataset

# Trains and stores the model, plots the learning curves
def training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))

    train_dataset, valid_dataset = setup_data_splits_and_transforms()

    # Define the model class
    model = models.resnet18(weights=None).to(device)
    model.fc = nn.Linear(model.fc.in_features, 8)

    # Most of the hyperparams were taken from the ResNet paper https://arxiv.org/abs/1512.03385
    # Final setup:
    batch_size = 256
    momentum = 0.9
    weight_decay = 0.0001
    lr = 0.1
    n_epochs = 40
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size)
    val_loader = DataLoader(valid_dataset, batch_size)

    tr_losses, val_losses = fit(n_epochs, model, loss_fn, optimizer, train_loader, val_loader, device, MODEL_PATH)

    plot_learning_curves(tr_losses, val_losses)

if __name__ == "__main__":
    training()
