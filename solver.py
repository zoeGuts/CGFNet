import os
import math
import sys
import json
import time
import shutil
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import RandomErase, AverageMeter, accuracy
from models.main_models import ResNet18, ResNet34, ResNet50, ResNet101


class Solver():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:%d' % args.device_ids if torch.cuda.is_available() else "cpu")

        # save configuration file               
        config_path = f'{self.args.save_dir}/config.json'
        with open(config_path, 'w') as json_file:
            json.dump(vars(self.args), json_file)

        # model setting
        print("=> creating method: '{}', arch: '{}'".format(self.args.method, self.args.arch))
        if self.args.arch == 'resnet18':
            self.model = ResNet18(num_classes=args.numberofclass, att_type=args.method, reduction_ratio=args.reduction_ratio, 
                            beta=args.beta, method_version=args.method_version, channel_groups=args.channel_groups, 
                            sa_kernel_size=args.sa_kernel_size, no_spatial=args.no_spatial, no_lofct=args.no_lofct)      
        elif self.args.arch == 'resnet34':
            self.model = ResNet34(num_classes=args.numberofclass, att_type=args.method, reduction_ratio=args.reduction_ratio, 
                            beta=args.beta, method_version=args.method_version, channel_groups=args.channel_groups, 
                            sa_kernel_size=args.sa_kernel_size, no_spatial=args.no_spatial,  no_lofct=args.no_lofct) 
        elif self.args.arch == 'resnet50':
            self.model = ResNet50(num_classes=args.numberofclass, att_type=args.method, reduction_ratio=args.reduction_ratio, 
                            beta=args.beta, method_version=args.method_version, channel_groups=args.channel_groups, 
                            sa_kernel_size=args.sa_kernel_size, no_spatial=args.no_spatial,  no_lofct=args.no_lofct)
        elif self.args.arch == 'resnet101':
            self.model = ResNet101(num_classes=args.numberofclass, 
              att_type=args.method, reduction_ratio=args.reduction_ratio, beta=args.beta,
              method_version=args.method_version, channel_groups=args.channel_groups, 
              sa_kernel_size=args.sa_kernel_size, no_spatial=args.no_spatial,  no_lofct=args.no_lofct)

        # optimizer and schduler
        pg = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(pg, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # lf = lambda x: ((1 + math.cos(x * math.pi / self.args.epochs)) / 2) * (1 - self.args.lrf) + self.args.lrf  # cosine
        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        # weights init
        if not self.args.load_model is None:
            self.model.load_state_dict(torch.load(self.args.load_model), strict=True)
            print('load model from %s' % self.args.load_model)
        elif self.args.pretrained:
            assert os.path.exists(self.args.pretrained_weights), "weights file: '{}' not exist.".format(self.args.pretrained_weights)
            weights_dict = torch.load(self.args.pretrained_weights)
            weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
            # Remove weights about classification categories
            for k in list(weights_dict.keys()):
                if "classifier" in k or "fc.weight" in k or "fc.bias" in k:
                    del weights_dict[k]
            print(self.model.load_state_dict(weights_dict, strict=False))
        
        self.model= self.model.to(self.device)
        
        if not self.args.resume is None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.args.start_epoch = checkpoint['epoch']
                self.best_val_ac_score = checkpoint['best_val_ac_score']
                self.model.load_state_dict(checkpoint['state_dict'])
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        else:
            print('not load any resume model!')
        
        if self.args.freeze_layers:
            for name, para in self.model.named_parameters():
                # Except for the head, pre_logits, all other weights are frozen
                if "classifier" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
            print('Except head, pre_logits, all other weights are frozen!')
        
        torch.backends.cudnn.benchmark = True
        
        self.best_val_ac_score = 0
        self.data_processing()
        
    def data_processing(self):
        flip = True
        rotate = True
        rotate_min = -180
        rotate_max = 180
        rescale = True
        rescale_min = 0.8889
        rescale_max=1.0
        shear = True
        shear_min=-36
        shear_max=36
        translate = True
        translate_min=0
        translate_max=0
        random_erase = True
        random_erase_prob=0.5
        random_erase_sl=0.02
        random_erase_sh=0.4
        random_erase_r=0.3
        contrast = True
        contrast_min=0.9
        contrast_max=1.1
                
        self.image_transforms = {
              'train': transforms.Compose([
              transforms.Resize((256, 256), Image.BILINEAR),
              transforms.RandomRotation(degrees=45),
              # transforms.RandomAffine(  # This data augmentation may result in better model performance.
              #     degrees=(rotate_min, rotate_max) if rotate else 0,
              #     translate=(translate_min, translate_max) if translate else None,
              #     scale=(rescale_min, rescale_max) if rescale else None,
              #     shear=(shear_min, shear_max) if shear else None,
              # ),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            }  
       
    def train(self):
        # dataloader settings
        train_dataset = datasets.ImageFolder(self.args.train_dir, 
                                    transform=self.image_transforms['train'])
        train_sampler = None
        train_loader = DataLoader(train_dataset, 
                        batch_size=self.args.batch_size, 
                        shuffle=(train_sampler is None), 
                        pin_memory=self.args.pin_memory, 
                        num_workers=self.args.nw, 
                        sampler=train_sampler)  
        
        val_dataset = datasets.ImageFolder(self.args.val_dir, 
                                    transform=self.image_transforms['val'])
        val_loader = DataLoader(val_dataset, 
                        batch_size=self.args.batch_size, 
                        shuffle=False, 
                        pin_memory=self.args.pin_memory, 
                        num_workers=0)

        global_train_acc = []
        global_val_acc = []
        best_top1 = 0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            batch_time, data_time = AverageMeter(), AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            
            # switch to train mode            
            self.model.train()
            end = time.time()            
            for step, (image, label) in enumerate(train_loader):
                # set input
                data_time.update(time.time() - end)
                image, label = image.to(self.device), label.to(self.device)
                images = torch.autograd.Variable(image)
                labels = torch.autograd.Variable(label)
                
                lr = adjust_learning_rate(self.optimizer, epoch, self.args, batch=step,
                                    nBatch=len(train_loader), method=self.args.lr_type)
                
                self.optimizer.zero_grad()
                pred, loss = self.model(images, labels)
                    
                losses.update(loss.item(), image.size(0))
                
                pred1, pred5 = accuracy(pred.data, labels, topk=(1, 5))
                top1.update(pred1.item(), image.size(0))
    
                loss.backward()
                self.optimizer.step()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                if (step + 1) % self.args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}] | '
                        'LR :{lr:.6f} | '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                        'Prec {top1.val:.4f} ({top1.avg:.4f}) | '.format(
                            epoch, step + 1, len(train_loader), 
                            lr=lr, 
                            batch_time=batch_time,
                            data_time=data_time, 
                            loss=losses, 
                            top1=top1))
                    global_train_acc.append(top1.avg)
                
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)

            # self.scheduler.step()
            
            val_loss, val_ac_score, val_time, val_acc = self.val(val_loader)
            
            # remember best prec@1 and save checkpoint
            global_val_acc.append(val_acc)
            if val_acc > best_top1:
                best_top1 = val_acc
            if best_top1 > 95:
                torch.save(self.model.state_dict(), '{}/{}_{}.pth'.format(self.args.save_dir, self.args.expname, best_top1))
            is_best = val_ac_score > self.best_val_ac_score
            self.best_val_ac_score = max(val_ac_score, self.best_val_ac_score)
            
            save_checkpoint({
            'epoch': epoch + 1,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'best_val_ac_score': self.best_val_ac_score,
            'optimizer': self.optimizer.state_dict(),
            }, is_best, self.args.expname)

            print('train_time: %.4f - train_loss %.4f - train_acc %.4f - '
              'val_time: %.4f - val_loss %.4f - val_ac_score %.4f - '
              'best_val_acc %.4f' % (batch_time.sum, losses.avg, top1.avg, 
                                     val_time, val_loss, val_ac_score, 
                                     self.best_val_ac_score))
    
        if self.args.plot_acc_curv:
            ratio = len(train_dataset) / self.args.batch_size / self.args.print_freq
            # ratio = int(ratio)+1
            ratio = int(ratio)
            show_acc_curv(ratio, global_train_acc, global_val_acc, self.args.expname)
        
    def val(self, val_loader):
        criterion = torch.nn.CrossEntropyLoss()        

        # test the top-1 acc
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        correct = 0

        net = self.model
        
        net.eval()
        end = time.time()
        with torch.no_grad():
            for step, (image, label) in enumerate(val_loader):
                image, label = image.to(self.device), label.to(self.device, non_blocking=True)
                images = torch.autograd.Variable(image)
                labels = torch.autograd.Variable(label)

                pred = net(images)  # [B*times_aug, num_cls]
                loss = criterion(pred, labels)
                losses.update(loss.item(), image.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                pred1, pred5 = accuracy(pred.data, labels, topk=(1, 5))
                top1.update(pred1.item(), image.size(0))

                _, pre = torch.max(pred.data, 1)
                correct += (pre == labels).sum()

                if (step + 1) % self.args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec {top1.val:.4f} ({top1.avg:.4f})'.format(
                        step + 1, len(val_loader), 
                        batch_time=batch_time, 
                        loss=losses, 
                        top1=top1))
            acc = correct.item() * 100. / (len(val_loader.dataset))

        return losses.avg, top1.avg, batch_time.sum, acc


def save_checkpoint(state, is_best, prefix):
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar' % prefix)


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                        nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr, decay_rate = args.lr, 0.1
        if epoch >= args.epochs * 0.75:
            lr *= decay_rate ** 2
        elif epoch >= args.epochs * 0.5:
            lr *= decay_rate
    elif method == 'step30':
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def show_acc_curv(ratio, global_train_acc, global_test_acc, prefix):
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc

    test_x = train_x[ratio - 1::ratio]
    test_y = global_test_acc

    plt.title('ResNet50 ACC')
    plt.plot(train_x, train_y, color='green', label='training accuracy')
    plt.plot(test_x, test_y, color='red', label='testing accuracy')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('accs')
    plt.savefig('./fig/acc_curv_' + prefix + '.jpg')
    # plt.show()
