from argparse import ArgumentParser
import json
import logging
import datetime
import os.path as osp

import torch
from addict import Dict
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn

from dataset import get_dataloaders
from resnet import ResNet18


class AverageMetr:
    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def avg(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return 0


class MyConfig(Dict):
    @classmethod
    def from_json(self, path) -> 'MyConfig':
        with open(path) as f:
            loaded_json = json.load(f)
        return self(loaded_json)

    def update_form_args(self, args):
        for key, value in vars(args).items():
            if key in self.keys():
                if value is not None:
                    self[key] = value
                else:
                    continue
            self[key] = value


def configure_logging(path):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=osp.join(path, current_time), level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--config', type=str, help='config path')
    parser.add_argument('--resume', type=str, help='resuming checkpoint')
    parser.add_argument('--data', type=str, help='dataset path')
    parser.add_argument('--save_freq', type=int, help='frequency of saving checkpoints in epochs', default=5)
    parser.add_argument('--print_freq', type=int, help='step of printing statistics', default=10)
    parser.add_argument('--log_dir', type=str, help='directory for logging and saving checkpoints', default='log_dir')
    parser.add_argument('--test_freq', type=int, default=5, help='test every (test_freq) epoch')
    return parser


def compute_acc(output, target):
    _, top1_indices = output.topk(1, dim=1)
    correct = top1_indices.squeeze().eq(target)
    return torch.mean(torch.where(correct, 1., 0.))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(config):
    configure_logging(config.log_dir)
    train_dataloader, val_dataloader = get_dataloaders('cifar100', 'data', config.batch_size)
    model = ResNet18(num_classes=100)
    criterion = nn.CrossEntropyLoss()
    if config.get('resume', None) is not None:
        model.load_state_dict(torch.load(config.resume))
    if config.mode == 'test':
        val(model, val_dataloader, config, criterion)
    else:
        optimizer = Adam(model.parameters(), lr=config.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, threshold=0.001, patience=5)
        train(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, config, criterion)
    

def train(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, config, criterion):
    for epoch in range(config.epochs):
        logging.info(f'Training epoch {epoch}')
        avg_loss = AverageMetr()
        avg_acc = AverageMetr()
        model.train()
        for iter, (input_, target) in enumerate(train_dataloader):
            output = model(input_)
            loss = criterion(output, target)
            acc = compute_acc(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss.update(float(loss))
            avg_acc.update(float(acc))
            if iter % config.print_freq == 0:
                print(f'{iter}/{int(len(train_dataloader))} iter Loss: {loss}({avg_loss.avg()}) Acc: {avg_acc.avg()} Lr: {get_lr(optimizer)}')
            break
        
        lr_scheduler.step(acc)
        if (iter + 1) % config.test_freq == 0:
            test_acc, test_loss  = val(model, val_dataloader, config, criterion)
            
        if (iter + 1) % config.save_freq == 0:
            logging.info('Saving model')
            torch.save(model.state_dict(), osp.join(config.log_dir, f'epoch_{epoch}.pth'))
        

def val(model, val_dataloader, config, criterion):
    avg_loss = AverageMetr()
    avg_acc = AverageMetr()
    model.eval()
    logging.info(f'Testing model')
    for iter, (input_, target) in enumerate(val_dataloader):
        output = model(input_)
        loss = criterion(output, target)
        acc = compute_acc(output, target)
        avg_loss.update(float(loss))
        avg_acc.update(float(acc))
        if iter % config.print_freq == 0:
            print(f'{iter}/{int(len(val_dataloader))} iter Loss: {loss}({avg_loss.avg()}) Acc: {avg_acc.avg()}')
    
    print(f'Test acc {avg_acc.avg()} loss {avg_loss.avg()}')
    return avg_acc.avg(), avg_loss.avg()


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    config = MyConfig.from_json(args.config)
    config.update_form_args(args)
    main(config)
