import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.resnet import resnet18
from models.wideresnet import WRN34_10

def pgd_attk(model, xent, dataloader, alpha, epsilon, num_iters, lower=0, upper=1):
    model.eval()
    total_correct = 0
    for (inputs, targets) in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        noise = torch.FloatTensor(inputs.shape).uniform_(-epsilon, epsilon).cuda()
        x = Variable(torch.clamp(inputs+noise, min=lower, max=upper), requires_grad=True)
        
        for _ in range(num_iters):
            x.requires_grad_()
            logits = model(x)
            loss = xent(logits, targets)
            loss.backward()
            grads = x.grad.data
            x = x.data.detach() + alpha*torch.sign(grads).detach()
            x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, min=lower, max=upper)
            
        with torch.no_grad():
            logits = model(x)
        total_correct += torch.argmax(logits, dim=1).eq(targets).sum().item()
        
    avg_acc = total_correct / len(dataloader.dataset)
    print('Test robust accuracy (PGD attack): %.4f' % avg_acc)
    return avg_acc

def natural(model, dataloader):
    model.eval()
    total_correct = 0
    for (inputs, targets) in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            logits = model(inputs)
            
        total_correct += torch.argmax(logits, dim=1).eq(targets).sum().item()
        
    avg_acc = total_correct / len(dataloader.dataset)
    print('Test accuracy (natural samples): %.4f' % avg_acc)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--model', type=str, default='WRN')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=2)
    parser.add_argument('--num_iters', type=int, default=20)
    parser.add_argument('--num_restart', type=int, default=1)
    args = parser.parse_args()
    
    print(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    test_dataset = datasets.CIFAR10(root='/root/mnt/datasets/data', train=False, download=True, transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.model == 'ResNet':
        model = nn.DataParallel(resnet18(num_classes=args.num_classes).cuda())
    elif args.model == 'WRN':
        model = nn.DataParallel(WRN34_10().cuda())
        
    state_dict = torch.load(args.checkpoint)['state_dict']
    model.load_state_dict(state_dict)

    xent = nn.CrossEntropyLoss().cuda()
    
    nat_acc = natural(model, test_dataloader)
    
    avg_robust_acc = 0
    for _ in range(args.num_restart):
        avg_robust_acc += pgd_attk(model, xent, test_dataloader, args.alpha/255, args.epsilon/255, args.num_iters, lower=0, upper=1)
    
    if args.num_restart > 1:
        print('Avg robust accuracy (%d restart): %.4f' % (args.num_restart, avg_robust_acc/args.num_restart))