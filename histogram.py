import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from utils import * 
from model import * 
from PIL import Image

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-r', '--load_params', type=str, default='models/pcnn_lr:0.00020_nr-resnet3_nr-filters50_219.pth',
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=3, #5
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=50, #160
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-b', '--batch_size', type=int, default=1, # 64
                    help='Batch size during training per GPU')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)
# assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
# writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

if 'mnist' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()

if args.load_params:
    load_part_of_model(model, args.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')



torch.cuda.synchronize()
model.eval()
test_loss = 0.
with torch.no_grad():
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.to('cuda')
        output = model(input)
        loss = loss_op(input, output)
        test_loss += loss.data.item()
        del loss, output
        deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
        progress_bar(batch_idx, len(testloader), 'BPD: %.3f'
                % (test_loss/ deno))

    

