from torchvision import datasets, transforms
import torch

def calculate_mean_std(root):
    # Load your dataset
    dataset = datasets.ImageFolder(root=root, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((48, 36))]))

    # Calculate mean and standard deviation
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    mean = data[0].mean(dim=[0, 2, 3])
    std = data[0].std(dim=[0, 2, 3])

    return mean, std

def custom_transform(mean, std, split: str = 'train',):
    if split == 'train':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.GaussianBlur(kernel_size=3),
            transforms.Normalize(mean, std),
        ])
    elif split == 'val':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif split == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Invalid split name: {split}")