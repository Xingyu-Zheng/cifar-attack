import torch
import argparse
from torch.utils.data import DataLoader
import imageio
from  torchvision import utils as vutils

from datasets import victim_sort_dataset, victim_zfill_dataset, victim_iamges_dataset
from models import REGISTRY_MODEL
from utils import gaussian_kernel, my_clip, merge_1, merge_2, mkdir

import torchattacks



def attack(models, test_loader, device, attacker, eps, save=False, all=False, merge=0, attackers=[], weights=[]):
    # model.eval()
    corrects = [0 for _ in models]
    total = 0
    pic = 0
    for images, labels in test_loader:
        gaussian_smoothing = gaussian_kernel(device, kernel_size=5, sigma=1, channels=3)
        images, labels = images.to(device), labels.to(device)
        gauss_images = gaussian_smoothing(images)
        # gauss_images = my_clip(images, gauss_images, eps)
        # gauss_images = images
        if attacker is not None or merge != 0:
            if merge == 0:
                images = attacker(gauss_images, labels, gauss_images).to(device)
                # images = attacker(images, labels, gauss_images).to(device)
            elif merge == 1:
                images = merge_1(attackers, weights, images, labels, gauss_images)
            elif merge == 2:
                images = merge_2(attackers, images, labels, gauss_images)
        # with torch.no_grad():
        #     for i in range(len(models)):
        #         outputs = models[i](images)
        #         _, predicted = torch.max(outputs.data, 1)
        #         corrects[i] += (predicted == labels.to(device)).sum()
        #         # print(labels)
        #         # print(predicted)
        
        if save == True:
            for img in images:
                if pic % 1000 == 0 or all:
                    img_path = "../results/images/images/" + str(pic) + ".png"
                    vutils.save_image(img.cpu(), img_path, normalize=False)
                pic += 1
        
        total += labels.size(0)
    print()
    for i in range(len(models)):
        print('model' + str(i) + ' accuracy: %.2f %%' % (100 * float(corrects[i]) / total))

def main():

    mkdir('../results/images/images/')

    # Training settings
    parser = argparse.ArgumentParser(description='Attack Model')
    parser.add_argument('--load_path', type=str, default='../weights')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # model_names = ["resnet18", "vgg16", 'pgd/resnet18']
    # model_names = ["resnet18", "vgg16", 'mobilenetv2', 'wrn16_4']
    model_names = ["resnet18", "vgg16"]
    models = []
    for name in model_names:
        model = REGISTRY_MODEL[name.split('/')[-1]]()
        model.load_state_dict(torch.load(args.load_path + '/' + name + '.pth', map_location=device))
        model = model.to(device)
        model.eval()
        models.append(model)
    # test_loader = DataLoader(victim_sort_dataset(), args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(victim_zfill_dataset(), args.batch_size, shuffle=False, num_workers=args.num_workers)
    # test_loader = DataLoader(victim_iamges_dataset(), args.batch_size, shuffle=False, num_workers=args.num_workers)

    attacker = None
    eps = 40/255
    alpha = 1.2/255
    steps = 40
    # attacker = torchattacks.FGSM(models[0], eps=eps)
    # attacker = torchattacks.MIFGSM(models[2], eps=eps, alpha=alpha, steps=steps)
    attacker = torchattacks.VNIFGSM(models[0], eps=eps, alpha=alpha, steps=steps)
    # attacker = torchattacks.DIFGSM(models[0], eps=eps, alpha=alpha, steps=steps, random_start=True)
    attacker = torchattacks.DIFGSM(models[0], eps=eps, alpha=alpha, steps=steps, random_start=True, models=models)

    attackers = []
    for i in [0, 1]:
        # attackers.append(torchattacks.MIFGSM(models[i], eps=eps, alpha=alpha, steps=steps, decay=1.0))
        attackers.append(torchattacks.DIFGSM(models[i], eps=eps, alpha=alpha, steps=steps, random_start=True))
    # attacker = torchattacks.MultiAttack(attackers)
    
    # attack(models, test_loader, device, attacker, save=False, all=True)
    attack(models, test_loader, device, attacker, eps, save=True, all=True)
    # attack(models, test_loader, device, attacker, eps, save=False, all=True, merge=1, attackers=attackers, weights=[0.3, 0.4, 0.3])
    # attack(models, test_loader, device, attacker, eps, save=True, all=True, merge=1, attackers=attackers, weights=[0.5, 0.5])
    # attack(models, test_loader, device, attacker, eps, save=True, all=True, merge=2, attackers=attackers)


if __name__ == '__main__':
    main()
