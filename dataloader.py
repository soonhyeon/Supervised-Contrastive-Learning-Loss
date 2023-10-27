import torch 
from torchvision import transforms, datasets


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
def cifar10_loader():
    # construct data loader
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(root='./',
                                      transform=TwoCropTransform(train_transform),
                                      download=True)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True, sampler=train_sampler)

    return train_loader