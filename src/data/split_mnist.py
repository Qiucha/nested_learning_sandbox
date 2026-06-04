import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_split_mnist(batch_size=64, data_dir='./data', seed=42):
    """
    Generates 5 distinct tasks for Split-MNIST.
    Tasks are created by randomly shuffling the 10 classes and dividing them into 5 pairs.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    tasks_train, tasks_test, task_classes = [], [], []
    
    rng = np.random.default_rng(seed)
    all_classes = rng.permutation(10)
    
    for t in range(5):
        classes = all_classes[t * 2 : t * 2 + 2].tolist()
        task_classes.append(classes)
        
        # Isolate indices for the current task's classes
        train_idx = np.isin(train_dataset.targets.numpy(), classes)
        tasks_train.append(DataLoader(
            Subset(train_dataset, np.where(train_idx)[0]), 
            batch_size=batch_size, 
            shuffle=True
        ))
        
        test_idx = np.isin(test_dataset.targets.numpy(), classes)
        tasks_test.append(DataLoader(
            Subset(test_dataset, np.where(test_idx)[0]), 
            batch_size=batch_size, 
            shuffle=False
        ))
        
    return tasks_train, tasks_test, task_classes