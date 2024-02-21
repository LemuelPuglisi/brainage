import os
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai.data import PersistentDataset
from monai import transforms
from monai.networks.nets import DenseNet
from monai.utils import set_determinism
from tqdm import tqdm


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--cache_path',  type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--epochs',      type=int, default=100)
    args = parser.parse_args()

    train_transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=1.4, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=(130, 130, 130), mode='minimum', keys=['image']),
        transforms.RandGaussianNoiseD(prob=0.8, keys=['image']),
        transforms.Rand3DElasticD(sigma_range=(0.01, 1), magnitude_range=(0, 1),
                            prob=0.8, rotate_range=(0.18, 0.18, 0.18),
                            translate_range=(4, 4, 4), scale_range=(0.10, 0.10, 0.10),
                            spatial_size=None, padding_mode="border", keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

    valid_transforms_fn =  transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=1.4, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=(130, 130, 130), mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])


    df = pd.read_csv(args.dataset_csv)
    train_data = df[(df.split == 'train') & (df.diagnosis == 1)].to_dict(orient='records')
    valid_data = df[(df.split == 'valid') & (df.diagnosis == 1)].to_dict(orient='records')

    trainset = PersistentDataset(train_data, transform=train_transforms_fn, cache_dir=args.cache_path)
    validset = PersistentDataset(valid_data, transform=valid_transforms_fn, cache_dir=args.cache_path)
    
    train_loader = DataLoader(dataset=trainset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True)

    valid_loader = DataLoader(dataset=validset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True)
    
    loaders = { 'train': train_loader, 'valid': valid_loader }

    net = DenseNet(3,1,1, dropout_prob=0.2).to(DEVICE)
    net.load_state_dict(torch.load('models/densenet.pth'))
    optim = torch.optim.AdamW(net.parameters(), lr=1e-4)
    optim.load_state_dict(torch.load('models/optim.pth'))
    scaler = GradScaler()

    writer = SummaryWriter()
    
    total_counters = { 'train': 0, 'valid': 0 }

    for epoch in range(args.epochs):
        for mode in ['train', 'valid']:
            epoch_losses = []
            loader = loaders[mode] 
            net.train() if mode == 'train' else net.eval()
            
            for batch in tqdm(loader):
                
                with torch.set_grad_enabled(mode == 'train'):
                    with autocast(enabled=True):
                        y_true = batch['age'].to(DEVICE).view(-1, 1).float()
                        y_pred = net(batch['image'].to(DEVICE).float()).float()

                    if mode == 'train':
                        loss = F.l1_loss(y_pred, y_true, reduction='mean')
                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()
                        optim.zero_grad(set_to_none=True)
                        writer.add_scalar(f'{mode}/loss', loss.item(), global_step=total_counters[mode])
                        epoch_losses.append(loss.item())

                    else:
                        mae = torch.abs(y_pred - y_true).mean()
                        writer.add_scalar(f'{mode}/mae', mae.item(), global_step=total_counters[mode])
                        epoch_losses.append(mae.item())

                    total_counters[mode] += 1

            # log the epoch loss / mae
            writer.add_scalar(f'{mode}/epoch_avg', sum(epoch_losses) / len(epoch_losses), global_step=epoch)

        # Save the model after each epoch.
        if epoch > 5:
            torch.save(net.state_dict(), os.path.join(args.output_path, f'densenet-{epoch}.pth'))
            torch.save(optim.state_dict(), os.path.join(args.output_path, f'optim-{epoch}.pth'))
