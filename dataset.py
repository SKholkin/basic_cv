from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataloaders(dataset_name, root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if dataset_name == 'cifar10':
        train_dataset, val_dataset = CIFAR10(root, train=True, download=True, transform=transform), CIFAR10(root, train=False, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        train_dataset, val_dataset = CIFAR100(root, train=True, download=True, transform=transform), CIFAR100(root, train=False, download=True, transform=transform)
    else:
        raise ValueError("Wrong dataset name")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader
