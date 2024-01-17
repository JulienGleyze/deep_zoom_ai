import os
import cv2
from multiprocessing import Pool
import numpy as np
from tqdm.auto import tqdm

import argparse

def patch_image(input_images_folder, img_name, hr_out_folder, blur_out_folder, chunk_size=64, blur_kernel=None):
    
    img_path = os.path.join(input_images_folder, img_name)

    img = cv2.imread(img_path)

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    width, height, _ = img.shape
    
    num_chunks_x = height // chunk_size
    num_chunks_y = width // chunk_size
    
    for i in range(num_chunks_x):
        for j in range(num_chunks_y):

            left = i * chunk_size
            top = j * chunk_size
            right = left + chunk_size
            bottom = top + chunk_size
            
            hr_patch = img[top:bottom, left:right]
            blur_patch = cv2.filter2D(hr_patch, -1, blur_kernel)[::2,::2] # divise resolution by 2
            hr_patch_path = os.path.join(hr_out_folder, f"chunk_{i}_{j}_crop_{img_name}")
            blur_patch_path = os.path.join(blur_out_folder, f"chunk_{i}_{j}_crop_{img_name}")

            cv2.imwrite(hr_patch_path, hr_patch)
            cv2.imwrite(blur_patch_path, blur_patch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=None, type=str)
    parser.add_argument('--hr_dir', default=None, type=str)
    parser.add_argument('--blur_dir', default=None, type=str)

    args = parser.parse_args()

    input_images_folder = args.input_dir
    hr_output_images_folder = args.hr_dir
    blur_output_images_folder = args.blur_dir
    
    blur_kernel = np.genfromtxt('PSF_E10x2.csv', delimiter=';')

    if hr_output_images_folder == None:
        hr_output_images_folder = "hr_patch"
    
    if blur_output_images_folder == None:
        blur_output_images_folder = "blur_patch"
    
    os.makedirs(hr_output_images_folder, exist_ok=True)
    os.makedirs(blur_output_images_folder, exist_ok=True)
    input_images = [f for f in os.listdir(input_images_folder) if f.endswith(('.png'))]

    with Pool() as p:
        p.starmap(patch_image, [(input_images_folder, 
                                   img, 
                                   hr_output_images_folder,
                                   blur_output_images_folder, 256, blur_kernel) for img in input_images[:-10]])