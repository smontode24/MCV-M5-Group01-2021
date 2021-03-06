
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split


def get_training_datasets(img_size):
    traindir = "/home/group01/mcv/datasets/MIT_split/train"
    data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Scale(size = img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet values
        ])
    )
    
    val_samples = int(0.2 * len(data))
    train_dataset, val_dataset = random_split(data, [len(data)-val_samples, val_samples])
    print(f"Dataset details: MIT_split\n> Training: {len(train_dataset)} samples\n> Validation: {len(val_dataset)} samples\n")
    return train_dataset, val_dataset



def get_testing_dataset(img_size):
    test_dataset = datasets.ImageFolder("/home/group01/mcv/datasets/MIT_split/test",
        transforms.Compose([
            transforms.ToTensor(),
            #transforms.Scale(size = img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet values
        ]))
    print(f"Dataset details: MIT_split\n> Testing: {len(test_dataset)}")
    return test_dataset