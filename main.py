import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from models.resnet import resnet18
from models.wideresnet import WRN34_10

def ga_pgd(model, xent, inputs, targets, alpha, epsilon, num_iters, lower=0, upper=1):
    noise = torch.FloatTensor(inputs.shape).uniform_(-epsilon, epsilon).cuda()
    inputs = torch.clamp(inputs+noise, min=lower, max=upper)
    x = Variable(inputs, requires_grad=True)
    kappa = torch.zeros(inputs.size(0)).cuda()
    for _ in range(num_iters):
        x.requires_grad_()
        logits = model(x)
        flag = torch.argmax(logits, dim=1).eq(targets)
        kappa += flag
        
        loss = xent(logits, targets)
        loss.backward()
        grads = x.grad.data
        
        x = x.data.detach() + alpha*torch.sign(grads).detach()
        x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
        x = torch.clamp(x, min=lower, max=upper)
    
    return x, kappa
        
def training(epoch, model, dataloader, optimizer, alpha, epsilon, num_iters, lam, tb, lower, upper, burn_in_period):
    total = 0
    total_loss = 0
    total_correct = 0
    
    model.train()
    tanh = nn.Tanh()
    xent = nn.CrossEntropyLoss()
    for batch_ind, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        x, kappa = ga_pgd(model, xent, inputs, targets, alpha, epsilon, num_iters, lower, upper)
        
        weight = (1 + tanh(lam + 5*(1 - 2*kappa/num_iters))) / 2
        weight = weight * len(weight) / torch.sum(weight)
        
        logits = model(x)
        if epoch > burn_in_period:
            log_softmax = torch.log_softmax(logits, dim=1)
            target_probs = torch.sum(torch.eye(logits.size(1))[targets].cuda() * log_softmax, dim=1)
            loss = -torch.sum(weight*target_probs) / inputs.size(0)
        else:
            loss = xent(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct = torch.argmax(logits, dim=1).eq(targets).sum().item()
        total_correct += correct
        total_loss += loss.item()
        total += inputs.size(0)
        
        if batch_ind % 100 == 0:
            tb.add_scalar('train/acc', total_correct/total, (len(dataloader)*epoch)+batch_ind)
            tb.add_scalar('train/loss', total_loss/total, (len(dataloader)*epoch)+batch_ind)
            print('%d epoch [%d/%d] | loss: %.4f (avg: %.4f) | acc: %.4f (avg: %.4f)' % (epoch, batch_ind, len(dataloader), loss.item(), total_loss/total, correct/inputs.size(0), total_correct/total))
            
def evaluation(epoch, model, dataloader, alpha, epsilon, num_iters, lower, upper):
    model.eval()
    total_correct_nat = 0
    total_correct_adv = 0
    xent = nn.CrossEntropyLoss()
    for samples in dataloader:
        inputs, targets = samples[0].cuda(), samples[1].cuda()
        x = Variable(inputs, requires_grad=True)
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
            logits_nat = model(inputs)
            logits_adv = model(x)
        
        total_correct_nat += torch.argmax(logits_nat.data, dim=1).eq(targets.data).cpu().sum()
        total_correct_adv += torch.argmax(logits_adv.data, dim=1).eq(targets.data).cpu().sum()
        
    print('Validation | nat acc: %.4f | adv acc: %.4f ' % (total_correct_nat / len(dataloader.dataset), total_correct_adv / len(dataloader.dataset)))
    return (total_correct_nat / len(dataloader.dataset)).item(), (total_correct_adv / len(dataloader.dataset)).item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--model', type=str, default='WRN')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=2)
    parser.add_argument('--num_iters', type=int, default=10)
    parser.add_argument('--lam', type=int, default=-1)
    parser.add_argument('--burn_in_period', type=int, default=60)
    args = parser.parse_args()
    
    print(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.checkpoint, exist_ok=True)
    
    tb_filename = os.path.join(args.checkpoint, 'logs')
    if os.path.exists(tb_filename):    shutil.rmtree(tb_filename)
    tb = SummaryWriter(log_dir=tb_filename)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='/root/mnt/datasets/data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='/root/mnt/datasets/data', train=False, download=True, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.model == 'ResNet':
        model = nn.DataParallel(resnet18(num_classes=args.num_classes).cuda())
    elif args.model == 'WRN':
        model = nn.DataParallel(WRN34_10().cuda())

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = [30, 60]
    adjust_learning_rate = lr_scheduler.MultiStepLR(optimizer, scheduler, gamma=0.1)
    xent = nn.CrossEntropyLoss().cuda()
    best_acc_nat = 0
    best_acc_adv = 0
    
    for epoch in range(args.epochs):
        training(epoch, model, train_dataloader, optimizer, args.alpha/255, args.epsilon/255, args.num_iters, args.lam, tb, lower=0, upper=1, burn_in_period=args.burn_in_period)
        test_acc_nat, test_acc_adv = evaluation(epoch, model, test_dataloader, args.alpha/255, args.epsilon/255, args.num_iters, lower=0, upper=1)
        tb.add_scalar('test/acc_nat', test_acc_nat, epoch)
        tb.add_scalar('test/acc_adv', test_acc_adv, epoch)

        is_best_nat = best_acc_nat < test_acc_nat
        is_best_adv = best_acc_adv < test_acc_adv
        best_acc_nat = max(best_acc_nat, test_acc_nat)
        best_acc_adv = max(best_acc_adv, test_acc_adv)
        save_checkpoint = {'state_dict': model.state_dict(),
                           'best_acc_nat': best_acc_nat,
                           'test_acc_nat': test_acc_nat,
                           'best_acc_adv': best_acc_adv,
                           'test_acc_adv': test_acc_adv,
                           'optimizer': optimizer.state_dict()}
        torch.save(save_checkpoint, os.path.join(args.checkpoint, 'model'))
        if is_best_nat and is_best_adv:
            print('Current model achieved the best accuracy, so saved as the best model.')
            torch.save(save_checkpoint, os.path.join(args.checkpoint, 'best_model'))
        adjust_learning_rate.step()
        