# Image_Retrieval_with_CNN_baseline
Personal project, implementing Image Retrieval using CNN end-to-end. Baseline level.  

I suggest to use this repository as reference **for the workflow only**.  
Due to the fact that, the system's performance is only at the baseline level.  

## Blog  

Here is my blog about How Image Retrieval with CNN looks like.  

[*Image Retrieval with CNN - Baseline link*](https://aibenderel.github.io/2023/05/04/image_retrieval_cnn.html#thoughts-after-i-recovered-from-headache)

## Main Components  

### Dataset

1. I download **ImageNet-1k-medium-test (10k)** on Kaggle.  

The dataset has:  
- 10,000 images  
- 1000 folders (1000 categories)  
- each folder has 10 images  

2. I then take a small portion to make a smaller dataset for experimenting.  
From the original dataset, start from the top, I picked the first 100 folders.  
**Directory: Data**   

The experiment dataset has:  
- 1000 images  
- 100 folders (100 categories)
- each folder has 10 images 

<br/>

[*ImageNet-1k-medium-test (10k) link*](https://www.kaggle.com/datasets/kerrit/imagenet1kmediumtest-10k)

### Dataset Feature extraction  

**Script: data_extractor.py**  

- Will download the model automatically. **Directory: Model**.  
- Will export the data feature file. **File location: Feature/extract_1000_ft.pt**

### Image Retrieval  

**Script: image_retrieval.py**  

- Will get ***the last image*** in **directory: Input** as the target image.  
- Will show the result.  

**Result format**: Top 10 images from the dataset that are the most similar to the target image.  
The order is as follow:  
```
1  2  3  4
5  6  7  8
9  10 T

* 1 is the top 1 image, 10 is the top 10 image.  
* T is the target image.  
``` 

## Environment

Pytorch 2.0.0+cu118  
matplotlib 3.7.1


