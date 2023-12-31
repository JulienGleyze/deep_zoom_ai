import os
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
from torch.nn.functional import interpolate

def load_images_as_tensor(folder_path):
    images = []
    for filename in tqdm(os.listdir(folder_path), desc="Loading patches"):
        if filename.endswith(('.png')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
            images.append(image)
    return torch.stack([transforms.ToTensor()(img) for img in tqdm(images, desc="Building tensor")])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hr_dir', default=None, type=str)
    parser.add_argument('--blur_dir', default=None, type=str)

    args = parser.parse_args()

    hr_dir = args.hr_dir
    blur_dir = args.blur_dir
    
    if hr_dir==None:
        hr_dir = "hr_patch/"
    if blur_dir==None:
        blur_dir = "blur_patch/"
    
    
    hr_tensor = load_images_as_tensor(hr_dir)
    torch.save(hr_tensor, 'hr_dataset.pt')
    
    del hr_tensor
        
    blur_tensor = load_images_as_tensor(blur_dir)
    torch.save(blur_tensor, 'blur_dataset.pt')

