"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def loss_fn(out, labels):
    
    criterion = nn.MSELoss()   
    loss = criterion(out, labels)

    return loss

class SupvModel(nn.Module):
    def training_step(self, batch):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        images, labels = batch 
        images, labels = images.float(), labels.float()
        # print("label", labels.shape)
        images, labels = images.to(device), labels.to(device)
        out = self(images)                  
        out = out.float()
        
        loss = loss_fn(out, labels)         
        return loss
    
    def validation_step(self, batch):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        images, labels = batch 
        images, labels = images.float(), labels.float()
        images, labels = images.to(device), labels.to(device)
        out = self(images)                    
        loss = loss_fn(out, labels)           
        return {'loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        return {'train_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}],  train_loss: {:.4f}, ".format(epoch,  result['train_loss']))


class Net(SupvModel):
    def __init__(self, channels, xinput, yinput):
        
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(12))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))        
        self.fc = nn.Sequential(
             nn.Dropout(0.4),
            nn.Linear(512, 1024),
            nn.ReLU())
        self.fc1= nn.Sequential(
             nn.Dropout(0.4),
            nn.Linear(1024, 8))

        

    def forward(self, xb):
        
        #############################
        # Fill your network structure of choice here!
        #############################
        out = self.layer1(xb)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)        
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.fc1(out)
        # out = F.softmax(out)
        return out
