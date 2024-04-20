"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch
import argparse
from tqdm import tqdm
import torch.utils.data
from model import Model
from utils import ConverterForCTC
from dataset import AlignCollate, Num10kDataset

def validation(model, criterion, evaluation_loader, converter, args, device):
    """ Evaluation Function """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in evaluation_loader:
            images = images.to(device)
            labels, lengths = converter.encode(labels, args.batch_max_length)
            labels = labels.to(device)
            
            output = model(images)
            output = output.permute(1, 0, 2)  # CTC loss expects (seq_len, batch, num_classes)
            loss = criterion(output, labels, torch.tensor([output.size(0)] * output.size(1)), lengths)
            total_loss += loss.item()
            
            decoded_preds = converter.decode(output.cpu().detach().numpy(), lengths)
            decoded_labels = converter.decode(labels.cpu().detach().numpy(), lengths)
            
            total_correct += sum(pred == label for pred, label in zip(decoded_preds, decoded_labels))
            total_samples += len(labels)
    
    avg_loss = total_loss / len(evaluation_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def test(args, device):
    args.device = device
    print('\n'+'-' * 80)
    print('Device : {}'.format(device))
    print('-' * 80 + '\n')
    
    # Load the Validation Dataset
    AlignCollate_valid = AlignCollate(imgH=args.imgH, imgW=args.imgW)
    valid_dataset = Num10kDataset(args.valid_data)
    print("Loaded Validation Dataset, Length: ", len(valid_dataset))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=True,
        collate_fn=AlignCollate_valid, pin_memory=False)
    
    converter = ConverterForCTC(args.character)
    model = Model(args)
    model.load_state_dict(torch.load(args.saved_model))
    model.to(device)
    
    criterion = torch.nn.CTCLoss(blank=0, reduction='mean')
    
    val_loss, val_accuracy = validation(model, criterion, valid_loader, converter, args, device)
    
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='Maximum label length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    
    """ Model Architecture - DO NOT CHANGE """
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel for CNN')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel for CNN')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
    args = parser.parse_args()
    
    """ vocab / character number configuration """
    args.character = "0123456789"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)
    
    test(args, device)
