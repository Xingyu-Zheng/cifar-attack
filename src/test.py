import torch
import argparse
from torch.utils.data import DataLoader

from datasets import test_dataset
from models import REGISTRY_MODEL

import torchattacks



def model_test(model, test_loader, device, attacker=None):
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        if attacker is None:
            images = images.to(device)
        else:
            images = attacker(images, labels).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()
    print('accuracy: %.2f %%' % (100 * float(correct) / total))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--exp_path', type=str, default='../weights')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model = REGISTRY_MODEL[args.model]()
    model.load_state_dict(torch.load(args.load_path, map_location=device))
    model = model.to(device)
    test_loader = DataLoader(test_dataset(), args.batch_size, shuffle=False, num_workers=args.num_workers)
    # test_loader = DataLoader(victim_dataset(), args.batch_size, shuffle=False, num_workers=args.num_workers)

    # attacker = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=10)
    attacker = torchattacks.MIFGSM(model)
    model_test(model, test_loader, device)
    model_test(model, test_loader, device, attacker)


if __name__ == '__main__':
    main()
