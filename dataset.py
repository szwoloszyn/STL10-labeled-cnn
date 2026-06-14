import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import torchvision
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, subset, transform=None) -> None:
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]

        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.subset)


def get_datasets(crop_scale=(0.6,1)):
    transform = transforms.ToTensor()

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), # mirror 
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(96, scale=crop_scale),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
    ])

    dataset_train_base = torchvision.datasets.STL10(root='./data', split="train", download=True, transform=None)
    dataset_test_base = torchvision.datasets.STL10(root='./data', split="test", download=True, transform=None)
    dataset = ConcatDataset([dataset_train_base, dataset_test_base])

    train_size = 11000
    val_size = 1000
    test_size = 1000
    generator = torch.Generator().manual_seed(42)
    trainset_raw, valset_raw, testset_raw = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )

    trainset = MyDataset(trainset_raw, transform=transform_train)
    valset = MyDataset(valset_raw, transform=transform_valid)
    testset = MyDataset(testset_raw, transform=transform_valid)

    return trainset, valset, testset

def get_dataloaders(train=None, val=None, test=None, crop_scale=(0.6,1), batch_size=64, num_workers=6):
    
    if train is None or val is None or test is None:
        train, val, test = get_datasets(crop_scale)
    return(
        DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, prefetch_factor=6),
        DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    )
    


