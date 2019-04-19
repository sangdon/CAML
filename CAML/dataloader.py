import os, sys
import types

import torch as tc
from torchvision import datasets, transforms

def data_loader(root, batch_size, image_size, gray_scale=False):
    ld = types.SimpleNamespace()
    if gray_scale:
        tform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor()
            ])
    else:
        tform = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor()
            ])
    ld.train = tc.utils.data.DataLoader(
        datasets.ImageFolder(
            root+"/train", 
            transform=tform), 
        batch_size=batch_size, shuffle=True, num_workers=2)
    ld.val = tc.utils.data.DataLoader(
        datasets.ImageFolder(
            root+"/val", 
            transform=tform), 
        batch_size=batch_size, shuffle=True, num_workers=1)
    if os.path.exists(os.path.join(root, "test")):
        ld.test = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                root+"/test", 
                transform=tform), 
            batch_size=batch_size, shuffle=True, num_workers=1)
    return ld
