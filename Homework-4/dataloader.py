import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class DualDegradationDataset(Dataset):
    """Dataset for both rain and snow degraded images"""
    
    def __init__(self, root_dir, phase='train', crop_size=128):
        self.root_dir = root_dir
        self.phase = phase
        self.crop_size = crop_size
        
        # Paths setup 
        if phase in ['train', 'val']:
            self.clean_dir = os.path.join(root_dir, 'train', 'clean')
            self.degraded_dir = os.path.join(root_dir, 'train', 'degraded')
            
            # Create pairs of degraded and clean images 
            rain_degraded = sorted(glob.glob(os.path.join(self.degraded_dir, 'rain-*.png')))
            snow_degraded = sorted(glob.glob(os.path.join(self.degraded_dir, 'snow-*.png')))
            
            rain_clean = sorted(glob.glob(os.path.join(self.clean_dir, 'rain_clean-*.png')))
            snow_clean = sorted(glob.glob(os.path.join(self.clean_dir, 'snow_clean-*.png')))
            
            self.degraded_paths = rain_degraded + snow_degraded
            self.clean_paths = rain_clean + snow_clean
            
            # Flags: 0 for rain, 1 for snow 
            self.degradation_types = [0] * len(rain_degraded) + [1] * len(snow_degraded)
            
        elif phase == 'test':
            
            self.degraded_dir = os.path.join(root_dir, 'test', 'degraded')
            self.degraded_paths = sorted(glob.glob(os.path.join(self.degraded_dir, '*.png')))
            self.clean_paths = None
            self.degradation_types = None
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.degraded_paths)
    
    def __getitem__(self, idx):
        deg_path = self.degraded_paths[idx]
        degraded_img = Image.open(deg_path).convert('RGB')
        
        if self.phase in ['train', 'val']:
            clean_img = Image.open(self.clean_paths[idx]).convert('RGB')
            deg_type = self.degradation_types[idx]
            
            # Apply identical random crops for training 
            if self.crop_size and self.phase == 'train':
                i, j, h, w = transforms.RandomCrop.get_params(clean_img, (self.crop_size, self.crop_size))
                clean_img = TF.crop(clean_img, i, j, h, w)
                degraded_img = TF.crop(degraded_img, i, j, h, w)
            
            return {
                'degraded': self.normalize(self.to_tensor(degraded_img)),
                'clean': self.normalize(self.to_tensor(clean_img)),
                'degradation_type': deg_type,
                'degraded_path': os.path.basename(deg_path)
            }
        else:
            
            return {
                'degraded': self.normalize(self.to_tensor(degraded_img)),
                'degraded_path': os.path.basename(deg_path)
            }

def get_dataloaders(data_root, batch_size=4, val_split=0.1):
    """Factory function for train.py and inference.py"""
    full_dataset = DualDegradationDataset(data_root, phase='train')
    
    # Split training data into train and val
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    val_ds.dataset.phase = 'val' 

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    
    test_ds = DualDegradationDataset(data_root, phase='test')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader
