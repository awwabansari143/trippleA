"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch.nn as nn

class CNNModule(nn.Module):
    """ The CNN Model for feature extraction """
    def __init__(self, input_channel=1, output_channel=512):
        super(CNNModule, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, output_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.features(input)
        return output
