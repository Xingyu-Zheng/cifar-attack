"""
Models in cifar are based on https://github.com/kuangliu/pytorch-cifar, and models of mnist are modified on them.
Models in imagenet comes from torchvision 0.8.1.

Model family:
mnist: input 28x28, 1 channel
cifar: input 32x32, 3 channels
imagenet: input 224x224, 3 channels
"""

from . import cifar
from .generator import BaseGenerator

def get_family(dataset):
    if dataset in ["cifar10", "cifar100", "svhn", "gtsrb"]:
        return "cifar"
    if dataset in ["mnist", "kmnist", "fashionmnist"]:
        return "mnist"
    if dataset in []:
        return "imagenet"
    return ""


REGISTRY_MODEL = {
    "resnet18": cifar.resnet18,
    "resnet50": cifar.resnet50,
    "lenet5": cifar.lenet5,
    "wrn16_4": cifar.wrn16_4,
    "vgg16": cifar.vgg16,
    "mobilenetv2": cifar.mobilenetv2,
    'googlenet': cifar.googlenet,
    'inception_v3': cifar.inception_v3,
    'vit-': cifar.vit,
}

