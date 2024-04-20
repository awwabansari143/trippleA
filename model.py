"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch.nn as nn
from rnn import LSTM
from cnn import CNNModule

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        """ CNN Module """
        self.CNNModule = CNNModule(args.input_channel, args.output_channel)
        self.CNN_output = args.output_channel

        """ RNN Module """
        self.RNNModule = LSTM(self.CNN_output, args.hidden_size, args.hidden_size)
        self.RNN_output = args.hidden_size

        """ Prediction Module """
        self.Prediction = nn.Linear(self.RNN_output, args.num_class)

    def forward(self, input):
        # Input Shape: BatchSize x 1 x imgH x imgW
        
        # Pass input through the CNNModule and process
        cnn_output = self.CNNModule(input)
        batch_size, channels, height, width = cnn_output.size()
        cnn_output = cnn_output.permute(0, 3, 1, 2).contiguous()  # Reshape for RNN
        cnn_output = cnn_output.view(batch_size, width, -1)

        # Pass through the RNNModule
        rnn_output = self.RNNModule(cnn_output)

        # Pass through the Prediction Layer
        prediction = self.Prediction(rnn_output)

        return prediction
