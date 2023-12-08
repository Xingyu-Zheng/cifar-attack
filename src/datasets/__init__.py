from torchvision import datasets, transforms

train_transform_cifar = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
test_transform_cifar = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)



dataset_root = '../data/cifar10'

def train_dataset():
    return datasets.CIFAR10(root=dataset_root, train=True, transform=train_transform_cifar)
def test_dataset():
    return datasets.CIFAR10(root=dataset_root, train=False, transform=test_transform_cifar)
def victim_sort_dataset():
    return datasets.ImageFolder(root='../data/victims-sort', transform=test_transform_cifar)
def victim_zfill_dataset():
    return datasets.ImageFolder(root='../data/victims-zfill', transform=test_transform_cifar)
def victim_iamges_dataset():
    return datasets.ImageFolder(root='../results/images', transform=test_transform_cifar)
