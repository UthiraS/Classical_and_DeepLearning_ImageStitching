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


def loss_fn(b_Pred, b_Patch):
    
    "Pixel Wise Photometric Loss"
    criterion = nn.L1Loss()
    
    b_Pred = torch.squeeze(b_Pred,1)
    # print("b_Pred.shape :",b_Pred.shape)
    # print("b_Patch.shape :",b_Patch.shape)
    loss = criterion(b_Pred, b_Patch)

    return loss

class UnSupvModel(nn.Module):
    def training_step(self, TCropABatch,TCropBBatch,TI1Batch, TImgABatch,TCornerBatch):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        TCropABatch= TCropABatch.to(device)
        TCropBBatch = TCropBBatch.to(device)
        TI1Batch = TI1Batch.to(device)
        TImgABatch = TImgABatch.to(device)
        TCornerBatch = TCornerBatch.to(device)        
        b_Pred = self(TCropABatch,TI1Batch, TImgABatch,TCornerBatch)                  
        
        
        loss = loss_fn(b_Pred, TCropBBatch.to(device))         
        return loss
    
    def validation_step(self, batch):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        images = batch 
        images = images.float()
        images = images.to(device)
        out = self(images)                    
        loss = loss_fn(out)           
        return {'loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        return {'train_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, train_loss: {:.4f}, ".format(epoch, result['val_loss'], result['train_loss']))

class TensorDLT(nn.Module):

        def __init__(self):

         super().__init__()

        # print("Inside Tensor DLT:")
        def forward(self, H4PT,CornerABatch):
            # a = torch.ones([32,768])
            #     b = torch.ones([32,512,768])
            
            batch_size = H4PT.size(0)
            # print("Batch_size :",batch_size)
            H = torch.ones([3,3],dtype=torch.double)
            H = torch.unsqueeze(H, 0)
            # print("H shape :",H.size())
            # Hi = torch.empty(batch_size,3,3)
            #H4PT #The size is batchsize*8*1
            #CornerABatch#The size is batchsize*8*1
            for H4pt,CornerA in zip(H4PT,CornerABatch): 
                
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                CornerB = CornerA + H4pt 
                # print("CornerA shape :",CornerA.size())
                # print("H4pt shape :",H4pt.size())
                A = []
                B = []
                for i in range(0,8,2): 
                    Ui = CornerA[i]
                    Vi = CornerA[i+1]
                    Uidash = CornerB[i]
                    Vidash = CornerB[i+1]
                    Ai = [[0, 0, 0, -Ui, Vi, -1, Vidash*Ui, Vidash*Vi],
                        [Ui, Vi, 1, 0, 0, 0, -Uidash*Ui, -Uidash*Vi]]
                    # A.append()
                    A.extend(Ai)

                    bi = [-Vidash,Uidash]
                    B.extend(bi)
                B= torch.tensor(B)
                B = torch.unsqueeze(B, 1)
                # print("B shape :",B.size())
                A = torch.tensor(A).to(device)
                B = (B).to(device)
                Ainv = torch.inverse(A)
                Ainv = Ainv.to(device)
                # print("Ainv shape :",Ainv.size())
                # print("A shape :",A.size())
                # print("B shape :",B.size())
                Hi = torch.matmul(Ainv, B)
                # print("Hi shape :",Hi.size())
                H33 = torch.tensor([1])
                # Hi =Hi.reshape(1,-1)
                # print("Hi shape :",Hi.size())
                # print("H shape :",H.size())
                # print("Hi :",Hi)
                Hi = torch.flatten(Hi)
                # print("Hi shape :",Hi.size())
                Hi = torch.cat((Hi,H33),0)
                # print("Hi shape :",Hi.size())
                Hi= Hi.reshape([3,3])
                # print("Hi shape :",Hi.size())

                
                # H_temp = H[None,:, :]
                # H_temp = torch.squeeze(H_temp, 0)
                # print(" H_temp :",H_temp.shape)
                # print(H_temp[None,:,:].size)
                
                H = torch.cat([H, torch.unsqueeze(Hi, 0)])
                # Hi = torch.cat(Hi, torch.unsqueeze(Hi, 0))
                # print(" H  :",H.shape)
                # print(" Hi  :",Hi.shape)
            # H = H[1:,:]
            # print("Final H  :",H.shape)
            # return H
            # print("Final Hi  :",Hi.shape)
            # print("Final H  :",H[1:65,:,:].shape)
            return H[1:65,:,:]

class STN(nn.Module):

            def __init__(self):

                super().__init__()

            def forward(self,TImgABatch,HMatrix):
                TImgABatch=TImgABatch.unsqueeze(1)
                TImgABatch = TImgABatch.to(torch.double)
                out = kornia.geometry.warp_perspective(TImgABatch, HMatrix, (128, 128), align_corners=True)

                return out
def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

class UnsupNet(UnSupvModel):
    def __init__(self,channels, xinput, yinput):
        
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
            nn.MaxPool2d(2))
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
            nn.ReLU(inplace=True),
            nn.MaxPool2d(6))        
        self.fc = nn.Sequential(
             nn.Dropout(0.4),
            nn.Linear(512, 1024),
            nn.ReLU())
        self.fc1= nn.Sequential(
             nn.Dropout(0.4),
            nn.Linear(1024, 8))

        self.tensorDLT=TensorDLT()
        self.stn=STN()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

        
        
   
                
    def forward(self, TCropABatch,TI1Batch, TImgABatch,TCornerBatch):
                
                #############################
                # Fill your network structure of choice here!
                #############################

                # Homography Network - Find H4PT between two images 
                out = self.layer1(TI1Batch)
                # print(TI1Batch.shape)
                out = self.layer2(out)
                # print(out.shape)
                out = self.layer3(out)
                # print(out.shape)
                out = self.layer4(out)
                # print(out.shape)
                out = self.layer5(out)
                # print(out.shape)
                out = self.layer6(out)
                # print(out.shape)
                out = self.layer7(out)
                # print(out.shape)
                out = self.layer8(out)        
                # print(out.shape)
                out = torch.flatten(out, 1)
                # print(out.shape)/
                out = self.fc(out)
                H4PT = self.fc1(out)

                #Tensor DLT - Find H matrix from H4PT and CornersA - Find H matrix
                HMatrix = self.tensorDLT(H4PT,TCornerBatch)
                HMatrix.requires_grad_()

                #STN - Find Warped Patch B from Hmatrix and Original Image
                b_Pred = self.stn(TImgABatch,HMatrix)



                

                
                
                return b_Pred
