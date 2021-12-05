import numpy as np
from dataloaders.dataloader import MyDataloader
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import torch

iheight, iwidth = 512, 512 # raw image size

class LIDARDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(LIDARDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (512,512)

    def train_transform(self, rgb, depth, target):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        s = int(512 * s)
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = float(torch.randint(0, 2, (1,)).item())

        tr = T.Compose([
            #T.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            #T.RandomRotation(abs(angle)),
            T.Resize((s, s), T.InterpolationMode.NEAREST),###add nearest
            T.CenterCrop(self.output_size),
            T.RandomHorizontalFlip(do_flip)
        ])
        rgb = TF.rotate(rgb, angle)
        rgb = tr(rgb)
        depth = TF.rotate(depth, angle)
        depth = tr(depth)
        target = TF.rotate(target, angle)
        target = tr(target)
        rgb = T.ColorJitter()(rgb)
        return rgb, depth, target


 
    def val_transform(self, rgb, depth, target):
        return rgb, depth, target
        