import torch
import os

from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode as IRM



class ImageNet_extract_1000(Dataset):
    
    def __init__(self, anno_file, image_dir, transform= None,
                 target_transform= None):
        
        with open(anno_file, "r") as file:
            labels = file.readlines()
            
        self.image_label = {}
        for string in labels:
            label_list = string.split()
            image = label_list[0]
            label = " ".join(label_list[1:])
            self.image_label[image] = label
            
        
        dir_names = os.listdir(image_dir)
        dir_paths = [
            os.path.join(image_dir, x) for x in dir_names]
        image_names = [
            os.listdir(x) for x in dir_paths]
        
        self.image_paths = []
        for path, name_list in zip(dir_paths, image_names):
            for image_name in name_list:
                image_path = os.path.join(path, image_name)
                self.image_paths.append(image_path)
        

        self.transform = transform
        self.target_transform = target_transform
       
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image_path = self.image_paths[idx]
        image = read_image(image_path, IRM.RGB)
        
        image_dir = os.path.dirname(image_path)
        dir_name = os.path.basename(image_dir)
        label = self.image_label.get(dir_name, -1)
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
        
