import torch

from utils import function
from dataset_custom import ImageNet_extract_1000

from torch.utils.data import DataLoader



#Hyperparameter

batch_size = 32

####



#Device

device = "cuda:0" if torch.cuda.is_available() else "cpu"

####



#Module

def extract_data_features(dls, model, device):
    
    model.eval()
    model.to(device)
    
    num_batches = len(dls)
    
    data_features = torch.empty(
        (0), dtype= torch.float32, device= device)
    
    for i, (images, labels) in enumerate(dls):
        images = images.to(device)
        
        with torch.no_grad():
            features = function.extract_feature(
                images, model)
            data_features = torch.cat(
                (data_features, features), dim= 0)
        
        
        print(
            f"Current batch: {i+1}/{num_batches}", 
            end= "\r")
        
    return data_features

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
        transform = transform)
    
    
    dls = DataLoader(
        data,
        batch_size = batch_size,
        shuffle = False)
    
    #! dls returns SAME ORDER of images everytime.
    
    ####
    
    
    
    #Model
    
    model = function.get_model()
    
    ####
    
    
    
    #Feature
    feature_PATH = "Feature/extract_1000_ft.pt"
    
    data_features = extract_data_features(
        dls, model, device)
    
    torch.save(data_features, feature_PATH)
    
    ####
    
    