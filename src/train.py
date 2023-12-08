import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from vit_pytorch import SimpleViT

from datasets import train_dataset, test_dataset
from models import REGISTRY_MODEL

class Model:
    def __init__(self, model, exp_path, batch_size=128, num_workers=4, epochs=120, lr=0.1, device=torch.device("cuda"), **kwargs):
        self.device = device
        self.exp_path = exp_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr

        self.modelname = model
        if model == 'vit':
            self.model = SimpleViT(
                image_size=32,
                patch_size=1,
                num_classes=10,
                dim=256,
                depth=2,
                heads=4,
                mlp_dim=128
            ).to(self.device)
        else:
            self.model = REGISTRY_MODEL[model]().to(self.device)
        


    def __call__(self, x):
        return self.model(x)
    
    def eval(self):
        self.model.eval()

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, dataset_train, dataset_test):
        criterion = nn.CrossEntropyLoss()

        opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs, last_epoch=-1)

        dataloader_train = DataLoader(dataset_train, self.batch_size, shuffle=True, num_workers=self.num_workers)
        dataloader_test = DataLoader(dataset_test, self.batch_size, shuffle=False, num_workers=self.num_workers)

        for epoch in range(1, self.epochs + 1):
            s = time.time()
            train_loss, train_acc = self.train_epoch(dataloader_train, opt, criterion)
            test_acc = self.test(dataloader_test)
            if sch:
                sch.step()
            e = time.time()
            time_epoch = e - s
            print(
                "Epoch: {} train_loss: {:.3f} train_acc: {:.2f}%, test_acc: {:.2f}% time: {:.1f}".format(
                    epoch, train_loss, train_acc * 100, test_acc * 100, time_epoch
                )
            )
            if epoch == self.epochs:
                torch.save(self.model.state_dict(), f"{self.exp_path}/{self.modelname}.pth")

    def train_epoch(self, data_loader, opt, criterion):
        """
        Train for 1 epoch
        """
        self.model.train()
        running_loss = correct = 0.0
        n_batches = len(data_loader)
        for (x, y) in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            pred = self.model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            pred_class = torch.argmax(pred, dim=-1)
            if y.ndim == 2:
                y = torch.argmax(y, dim=-1)
            correct += (pred_class == y).sum().item()

        loss = running_loss / n_batches
        acc = correct / len(data_loader.dataset)
        return loss, acc

    def test(self, data_loader):
        self.model.eval()
        correct = 0.0
        with torch.no_grad():
            for (x, y) in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.__call__(x)
                pred_class = torch.argmax(pred, dim=1)
                correct += (pred_class == y).sum().item()
            acc = correct / len(data_loader.dataset)
        return acc



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--exp_path', type=str, default='../weights')
    args = parser.parse_args()
    
    model = Model(
        model=args.model,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        exp_path=args.exp_path,
        )
    model.train(train_dataset(), test_dataset())


if __name__ == '__main__':
    main()