import random
import os
import sys
import json
import math
import time

import numpy as np
from torch.utils.data import Dataset

import torch.utils.data as data
import utils.utils_image as util

import torch
from torch.autograd import Variable
import nibabel as nb
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from patchify import patchify
import cv2


plt.switch_backend('agg')

np.random.seed(int(time.time()))  # Seed with the current time

def tensor_patchify(volume, patch_size=(4, 64, 64), step_size=(3, 54, 54), num_samples=4):
    B, C, D, H, W = volume.shape
    pD, pH, pW = patch_size
    sD, sH, sW = step_size

    # Compute number of patches along each axis
    num_patches_d = (D - pD) // sD + 1
    num_patches_h = (H - pH) // sH + 1
    num_patches_w = (W - pW) // sW + 1

    patches = []

    for sample in volume:
        # Extract patches
        for d in range(0, num_patches_d * sD, sD):
            for h in range(0, num_patches_h * sH, sH):
                for w in range(0, num_patches_w * sW, sW):
                    patch = sample[:, d:d + pD, h:h + pH, w:w + pW]
                    patch = torch.unsqueeze(patch, dim=0)
                    patches.append(patch)

    # Stack patches into a single tensor
    patches = torch.cat(patches, dim=0)

    return patches


class LRHRDataset(Dataset):

    """
        Args:
            first_hf_path  : Directory containing BraTS subfolders (BraTS2021_xxx).
            second_hf_path : Typically the same as first_hf_path, but you could vary.
            patch_size     : Tuple (D,H,W) specifying sub-volume shape to extract.
            step_size      : Not necessarily used in this minimal example, but included for reference.
            opt            : A dictionary of other possible options (if needed).
     """
    
    # first_hf_path, second_hf_path are the same in this experiment
    def __init__(self, first_hf_path, second_hf_path, patch_size, step_size, opt):
        self.first_hf_path = first_hf_path
        self.second_hf_path = second_hf_path

        self.patients = self.load_patients_(self.first_hf_path)
        self.data_len = len(self.patients)
        self.patch_size = patch_size
        self.step_size = step_size
        self.opt = opt
        self.gaussian_noise = np.zeros((len(self.patients), 1, 240, 240))
        self.gaussian_noise = self.gaussian_noise.astype(np.float32)


    def sample_normalize_(self, volume):
        volume = volume.astype(np.float32)
        volume = volume/volume.max()
        return volume

    def load_nb_(self, subject_paths):
        contrast_images = {}
        for contrast in subject_paths:
            image_nb = nb.load(subject_paths[contrast])
            image = image_nb.get_fdata()
            contrast_images[contrast] = np.expand_dims(self.sample_normalize_(image[:,:,::14].swapaxes(1, 2).swapaxes(1, 0)), axis=0)

        return contrast_images

    def load_patients_(self, path):
        patients = [f for f in os.listdir(path) if f.startswith('BraTS')]

        return patients

    # def add_gaussian_noise_(self, image, idx, mean=0, stddev=0.1):

    #     noise = np.random.normal(mean, stddev, image.shape)
    #     noisy_image = image + noise

    #     return noisy_image
    
    def add_gaussian_noise_(self, image, idx, mean=0, stddev=0.1):
 
        if np.all(self.gaussian_noise[idx] != 0):
            noise = self.gaussian_noise[idx]
 
        else:
            noise = np.random.normal(mean, stddev, (1, 240, 240))
            self.gaussian_noise[idx] = noise
 
        # noise_real = np.random.normal(mean, stddev, image.shape)  # N_r
        # noise_imag = np.random.normal(mean, stddev, image.shape)  # N_i
 
        # Compute Rician noise in the magnitude domain
        # noisy_image = np.sqrt((image + noise_real)**2 + noise_imag**2)
            # self.gaussian_noise[idx] = noise
 
        noise = np.expand_dims(noise, axis=0)  # Shape: (1, 1, 240, 240)
        noise = np.repeat(noise, 12, axis=1)  # Shape: (1, 12, 240, 240)
 
        # # noise = np.random.normal(mean, stddev, (1, 240, 240))
        
        noisy_image = image + noise
 
        return noisy_image

    def synthesize_(self, subject_paths, idx):
        HF_B = 3
        resolutions = [1, 2, 1, 2]
        magnetic_fields = np.linspace(50, 100, 12)
        noise_coeffs = np.linspace(6, 12, 7)

        magnetic_fields = list(magnetic_fields/1000)
        noise_coeffs = noise_coeffs/100

        random.shuffle(magnetic_fields)
        B = magnetic_fields[0]

        hf_contrast_images = {}
        lf_contrast_images = {}
        current_resolution = 1  # Original voxel resolution
        resolutions_copy = resolutions.copy()
        noise_coeffs_copy = noise_coeffs.copy()
        
        random.shuffle(resolutions_copy)
        new_resolution = resolutions_copy[0]
        zoom_factor = current_resolution / new_resolution

        random.shuffle(noise_coeffs_copy)
        noise_coeff = noise_coeffs_copy[0]
        


        for contrast in subject_paths:
            image_nb = nb.load(subject_paths[contrast])
            image = image_nb.get_fdata()[:,:,::14].swapaxes(1, 2).swapaxes(1, 0)
            # image_lf = image/((HF_B/B)**2)
            
            
            hf_contrast_images[contrast] = np.expand_dims(self.sample_normalize_(image), axis=0)
            # lf_contrast_images[contrast] = self.simulate_low_field_arnold(hf_contrast_images[contrast]).astype(np.float32)
            lf_contrast_images[contrast] = self.add_gaussian_noise_(hf_contrast_images[contrast], idx, stddev = noise_coeff).astype(np.float32) # adds incremental noise to images
            
        return lf_contrast_images, hf_contrast_images, B
    
    def sample_normalize_(self, volume):
        volume = volume.astype(np.float32)
        volume -= volume.min()
        volume = volume/volume.max()
        return volume 

    def simulate_low_field_arnold(self, volume, sigma=0.5000, noise_coef=2.5000, noise_coef2=1.5000, noise_coef3=0.5000):
        mask = (volume > 0).astype(np.float32)
        noise_range = noise_coef * volume[volume > 0].std()
        noise1 = np.random.uniform(size=volume.shape) - 0.5
        noise_add = noise_range * noise1
        noise_add = gaussian_filter(noise_add,0.5)
        img_noise = volume + noise_add

        img_gauss = gaussian_filter(img_noise,sigma)
        img_noise = img_gauss * mask

        noise_range = noise_coef2 * img_noise[img_noise > 0].std()
        noise2 = np.random.uniform(size=volume.shape) - 0.5
        noise_add = noise_range * noise2
        noise_add = gaussian_filter(noise_add,noise_coef3)
        img_noise = img_noise + noise_add
        img_noise = img_noise * mask

        return self.sample_normalize_(img_noise)
 
    def lf_data_gen_(self, path, patient_mrn, idx):
        flair_path = '{}/{}/{}_flair.nii.gz'.format(path, patient_mrn, patient_mrn)
        t1_path = '{}/{}/{}_t1.nii.gz'.format(path, patient_mrn, patient_mrn)
        t2_path = '{}/{}/{}_t2.nii.gz'.format(path, patient_mrn, patient_mrn)
        t1ce_path = '{}/{}/{}_t1ce.nii.gz'.format(path, patient_mrn, patient_mrn)

        lf_images, hf_images, B = self.synthesize_({'t1':t1_path, 't2':t2_path, 'flair':flair_path, 't1ce':t1ce_path}, idx)

        return lf_images, hf_images, B

    def get_hf_lf_pair_(self, first_hf_path, second_hf_path, patient_mrn, idx):
        flair_path = '{}/{}/{}_flair.nii.gz'.format(first_hf_path, patient_mrn, patient_mrn)
        t1_path = '{}/{}/{}_t1.nii.gz'.format(first_hf_path, patient_mrn, patient_mrn)
        t2_path = '{}/{}/{}_t2.nii.gz'.format(first_hf_path, patient_mrn, patient_mrn)
        t1ce_path = '{}/{}/{}_t1ce.nii.gz'.format(first_hf_path, patient_mrn, patient_mrn)
        seg_path = '{}/{}/{}_seg.nii.gz'.format(first_hf_path, patient_mrn, patient_mrn)
        first_hf_images = self.load_nb_({'t1':t1_path, 't2':t2_path, 'flair':flair_path, 't1ce':t1ce_path})

        image_nb = nb.load(seg_path).get_fdata()
        first_hf_images['seg'] = image_nb[:,:,::14].swapaxes(1, 2).swapaxes(1, 0)

        second_lf_images, second_hf_images, B = self.lf_data_gen_(second_hf_path, patient_mrn, idx)

        # first_hf_images and second_hf_images in 2D example are gonna be the same
        # select one of them as the groundtruth
        return first_hf_images, second_lf_images, second_hf_images, B

    def __len__(self):
        return self.data_len


    def __getitem__(self, index):
        
        """
            Must produce a single pair: 
            {
            'L': <torch.tensor>, 
            'H': <torch.tensor>, 
            'L_path': <str>, 
            'H_path': <str>
            }
            where each tensor is shape [C, H, W] in range [0,1].
        """
        patient_mrn = self.patients[index]
        first_hf_images, second_lf_images, second_hf_images, B = self.get_hf_lf_pair_(self.first_hf_path, self.second_hf_path, patient_mrn, index)

        # contrasts = ['t1', 't2', 'flair', 't1ce']
        
        # Can do multi-contrast but let's start with FLAIR
        contrasts = ['flair']

        tumor_map = first_hf_images['seg']
        tumor_coords = np.argwhere(tumor_map > 0)

        first_hf_images = [first_hf_images[c] for c in contrasts]
        first_hf_images = np.concatenate(first_hf_images, axis=0)

        second_lf_images = [second_lf_images[c] for c in contrasts]
        second_lf_images = np.concatenate(second_lf_images, axis=0)

        second_hf_images = [second_hf_images[c] for c in contrasts]
        second_hf_images = np.concatenate(second_hf_images, axis=0)

        
        ## later when you need patching
        '''
        selected_coord = tumor_coords[np.random.choice(tumor_coords.shape[0])]
        selected_coord = tumor_coords[0]

        if np.random.rand() >= 0.1:
            if selected_coord[0] + self.patch_size[0] >= first_hf_images.shape[1]:
                drange = range(0, first_hf_images.shape[1] - self.patch_size[0])

            else:
                drange = range(selected_coord[0], selected_coord[0] + self.patch_size[0])

            if selected_coord[1] + self.patch_size[1] >= first_hf_images.shape[2]:
                hrange = range(0, first_hf_images.shape[2] - self.patch_size[1])

            else:
                hrange = range(selected_coord[1], selected_coord[1] + self.patch_size[1])
            
            if selected_coord[2] + self.patch_size[2] >= first_hf_images.shape[3]:
                wrange = range(0, first_hf_images.shape[3] - self.patch_size[2])

            else:
                wrange = range(selected_coord[2], selected_coord[2] + self.patch_size[2])

        else:
            drange = range(max(selected_coord[0] - self.patch_size[0], 0), max(selected_coord[0] - self.patch_size[0], 0) + self.patch_size[0])
            hrange = range(max(selected_coord[1] - self.patch_size[1], 0), max(selected_coord[1] - self.patch_size[1], 0) + self.patch_size[1])
            wrange = range(max(selected_coord[2] - self.patch_size[2], 0), max(selected_coord[2] - self.patch_size[2], 0) + self.patch_size[2])

        dstart = drange[0]
        hstart = random.randint(hrange[0], hrange[1])
        wstart = random.randint(wrange[0], wrange[1])
        '''

        # config 14, 1 slice, whole frame
        ## remember to set patch size the same as the volume W and H
        dstart = 7
        hstart = 0
        wstart = 0

        first_hf_images = first_hf_images[:, dstart: dstart + self.patch_size[0], hstart:hstart+self.patch_size[1], wstart:wstart+self.patch_size[2]]
        second_lf_images = second_lf_images[:, dstart: dstart + self.patch_size[0], hstart:hstart+self.patch_size[1], wstart:wstart+self.patch_size[2]]
        second_hf_images = second_hf_images[:, dstart: dstart + self.patch_size[0], hstart:hstart+self.patch_size[1], wstart:wstart+self.patch_size[2]]

        # Loads of empty space in the volume. Crop out those regions
        first_hf_images = first_hf_images[:, :, 40:-40, 40:-40]# 40:-40, 40:-40
        second_lf_images = second_lf_images[:, :, 40:-40, 40:-40]# 40:-40, 31:-17
        second_hf_images = second_hf_images[:, :, 40:-40, 40:-40] # 40:-40, 31:-17
        
        '''
        We must modify the dataset to achieve HF-LF pair and their respective paths
        '''
        # 5) For a 2D pipeline (like default KAIR), we pick a single slice from the D dimension.
        #    Let's pick slice #0 for example. You can pick random if you want.
        lf_slice = second_lf_images[:, 0, :, :]  # shape [C, H', W']
        hf_slice = second_hf_images[:, 0, :, :]

        # 6) Scale from [0..1] floats up to [0..255] so KAIR's "uint2tensor3()" works properly.
        #    Typically, "imread_uint()" reads 8-bit images in [0..255]. So let's mimic that:
        lf_slice_255 = (lf_slice * 255.0).clip(0, 255).astype(np.uint8)
        hf_slice_255 = (hf_slice * 255.0).clip(0, 255).astype(np.uint8)
        # shape => [C, H, W], each in [0..255].

        # 7) Convert to HWC for KAIR's "uint2tensor3()", which expects an [H, W, C] array
        #    (where C = 3 for color or 1 for grayscale).
        #    If you want 3-channel for your single-channel volume, you can replicate channels.
        #    For a single channel, we'll do:
        lf_slice_hwc = lf_slice_255.transpose(1, 2, 0)  # shape => [H, W, C=1]
        hf_slice_hwc = hf_slice_255.transpose(1, 2, 0)

        # 9) Now convert to Torch tensors [C, H, W], normalized to [0..1].
        #    Using the KAIR utility: "uint2tensor3()" => shape [C, H, W], float in [0..1].
        img_L = util.uint2tensor3(lf_slice_hwc)
        img_H = util.uint2tensor3(hf_slice_hwc)

        # Construct LF/HF path for the .nii files 
        hf_nii_path = '{}/{}/{}_{}.nii.gz'.format(self.first_hf_path, patient_mrn, patient_mrn, contrasts[0])
        HF_path = hf_nii_path  
        LF_path = hf_nii_path
        
        # print(f'LR: {np.shape(img_L)}, HR: {np.shape(img_H)}')

        # Return in the KAIR-friendly dict format
        return {
            'L': img_L,            # Low-resolution (synthesized)
            'H': img_H,            # High-resolution (original)
            'L_path': LF_path,    # Low-Field path (second h-field path)
            'H_path': HF_path     # H-Field path (first h-field path)
        }
