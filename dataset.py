"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class Num10kDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.labels = []
        self.image_paths = []
        with open(os.path.join(root_dir, 'labels.txt'), 'r') as f:
            for line in f:
                image_name, label = line.strip().split('\t')
                self.image_paths.append(os.path.join(root_dir, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)

        return image, label

class AlignCollate(object):
    """
    Used for collating and padding the images in the batch (labels being taken care by ConverterForCTC).
    Returns aligned image tensors and corresponding labels.
    """
    def __init__(self, imgH=32, imgW=100, input_channel=1):
        self.imgH = imgH
        self.imgW = imgW
        self.input_channel = input_channel

    def __call__(self, batch):
        images, labels = zip(*batch)

        # Resize and pad images
        resized_images = []
        transform = transforms.Compose([
            transforms.Resize((self.imgH, self.imgW)),
            transforms.ToTensor()
        ])
        for image in images:
            resized_image = transform(image)
            resized_images.append(resized_image)

        # Stack images into a single tensor
        aligned_images = torch.stack(resized_images, dim=0)

        return aligned_images, labels
