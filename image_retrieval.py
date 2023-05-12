import os

import torch

from utils import function

from dataset_custom import ImageNet_extract_1000

import torchvision
from torchvision.io import read_image, ImageReadMode as IRM

import matplotlib.pyplot as plt


#Hyperparameter

batch_size = 32

####



#Device

device = "cuda:0" if torch.cuda.is_available() else "cpu"

####



#Module

def extract_input_feature(input_image, model, device):
    
    model.eval()
    model.to(device)
    
    input_image = input_image.to(device)
    
    with torch.no_grad():
        return function.extract_feature(input_image, model)

####



if __name__ == "__main__":
    
    #Transform
    
    transform = function.transform()
    
    ####
    
    
    
    #Data
    anno_PATH = "Data/ImageNet/LOC_synset_mapping.txt"
    image_dir_PATH = "Data/ImageNet/extract_1000"

    data = ImageNet_extract_1000(
        anno_PATH,
        image_dir_PATH,
        transform = transform
    )
    
    ####
    
    
    
    #Feature
    feature_PATH = "Feature/extract_1000_ft.pt"

    data_features = torch.load(feature_PATH)
    
    ####
    
    
    
    #Input
    input_PATH = "Input/"
    
    input_name = os.listdir(input_PATH)[-1]
    input_path = os.path.join(input_PATH, input_name)

    input_image = read_image(input_path, IRM.RGB)
    input_tfm = transform(input_image)
    
    ####
    
    
    
    #Model

    model = function.get_model()
    
    ####
    
    
    
    #Distance
    
    input_feature = extract_input_feature(
        input_tfm, model, device)

    distances = torch.cdist(
        input_feature, data_features, p= 2)
    
    distances, indices = torch.sort(distances, dim= 1)
    
    ####
    
    
    
    #Top_k
    
    idx_top_k = indices[0][:10]     #top 10
    
    img_top_k = [data[x][0] for x in idx_top_k]
    lbl_top_k = [data[x][1] for x in idx_top_k]
    img_path_top_k = [
        data.image_paths[x] for x in idx_top_k]
    
    ####
    
    
    
    #Info
    
    print(f"0. {input_path}\n----------")
    
    for i, (label, path) in enumerate(
        zip(lbl_top_k, img_path_top_k)):
        
        print(f"{i+1}. {path}\n{label}\n----------")
    
    ####
    
    
    
    #Show
    
    img_top_k.append(input_tfm)
    grid = torchvision.utils.make_grid(img_top_k , nrow= 4)
    grid = grid.permute(1, 2, 0)

    plt.imshow(grid)
    plt.show()
    
    ####
    
    
    