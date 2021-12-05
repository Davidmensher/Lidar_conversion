import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
from PIL import Image
import imageio
import cv2
import torch
from torchvision import transforms as T
from dataloaders.depth_map_utils import fill_in_multiscale

IMG_EXTENSIONS = ['.h5','.jpg'] #elad edit add .png

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def no_depth_completion(depth_image):
    h, w = depth_image.shape
    depth_image = cv2.copyMakeBorder(depth_image, h, h, w, w, cv2.BORDER_REFLECT)
    projected_depths = np.float32(depth_image / 256.0)
    final_depths, process_dict = fill_in_multiscale(projected_depths)
    final_depths = final_depths[h:-h, w:-w]
    return final_depths

def inpainting_fillna(im, value=None):
    h, w = im.shape
    im_ = cv2.copyMakeBorder(im, h, h, w, w, cv2.BORDER_REFLECT)
    if value is None:
        mask = np.isnan(im_).astype('uint8')
    else:
        mask = (im_ == value).astype('uint8')
    dst = cv2.inpaint(im_.astype('float32'), mask, 3, cv2.INPAINT_NS)
    res = dst[h:-h, w:-w]
    return res
    
    
def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join('datasets','rgb_preped') 
        print(d)
        if not os.path.isdir(d):
            print("HHHH")
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth


class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
   
    def __init__(self, root, type, sparsifier=None, modality='rgb', loader=h5_loader):
        classes, class_to_idx = find_classes(root)
        #imgs = make_dataset(root, class_to_idx)
        #start elad edit
        if type == 'train':
            imgs = [os.path.join("..","s2d","datasets","rgb_preped",i) for i in sorted(os.listdir("../s2d/datasets/rgb_preped"))[:16650]] #elad edit
        if type == 'val':
            imgs = [os.path.join("..","s2d","datasets","rgb_preped",i) for i in sorted(os.listdir("../s2d/datasets/rgb_preped"))[16650:]] #elad edit
        
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    
    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd


    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        #path, target = self.imgs[index]
        #rgb, depth = self.loader(path)
        #return rgb, depth
        
        
        ####elad edit start###

        #print(self.imgs[index])

        im_frame = Image.open(self.imgs[index])
        rgb = T.ToTensor()(im_frame)
    
        depth = os.path.join("..","s2d","datasets","lidar_preped",self.imgs[index][27:-4] + ".exr")

        depth = cv2.imread(depth,-1)
        depth = no_depth_completion(depth)
        depth = T.ToTensor()(depth)
        
        
        
        target = os.path.join("..","s2d","datasets","ipad_preped",self.imgs[index][27:-4] + ".exr")

        target = cv2.imread(target,-1)
        target = T.ToTensor()(target)

        
        return rgb, depth, target
        ####elad edit end

    
    def __getitem__(self, index): #elad edit
        
        rgb, depth, target = self.__getraw__(index)
        rgb, depth, target =  self.transform(rgb, depth, target)
        
        return torch.cat((rgb, depth),0), target


    def __len__(self):
        return len(self.imgs)

    