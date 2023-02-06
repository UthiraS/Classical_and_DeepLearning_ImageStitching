#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network import Net
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch

from torchvision import transforms as tf

# Don't generate pyc codes
sys.dont_write_bytecode = True


# def SetupAll():
#     """
#     Outputs:
#     ImageSize - Size of the Image
#     """
#     # Image Input Shape
#     ImageSize = [32, 32, 3]

#     return ImageSize


# def StandardizeInputs(Img):
#     ##########################################################################
#     # Add any standardization or cropping/resizing if used in Training here!
#     ##########################################################################
#     return Img

def createPatch(img,patchSize = 128,pert= 32):
    
    
    # img = cv2.resize(img, (320, 240), interpolation = cv2.INTER_AREA) 
    #         # if(img.shape[0]>240 and img.shape[1] >240):
    # img    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x,y = img.shape
    patch_size = 128
    pert = 16
    pert = 2* pert
    if x-pert-patch_size > 0 and y-pert- patch_size > 0:
        
        random_x = random.randint(0+pert, x-patch_size-pert)
        random_y = random.randint(0+pert, y-patch_size-pert)
        PatchA = img[random_x : random_x + patch_size, random_y : random_y + patch_size]
        PatchA_corners = np.array([[random_y, random_x],
                                    [random_y, random_x + patch_size],
                                    [random_y + patch_size, random_x + patch_size],
                                    [random_y + patch_size, random_x]])
        

        points1 = np.array([[0, 0],
                        [0, x],
                        [y, x],
                        [y, 0]]).reshape(-1,1,2)

        random_x_pt = np.array(random.sample(range(-pert, pert), 4)).reshape(4,1)
        random_y_pt = np.array(random.sample(range(-pert, pert), 4)).reshape(4,1)
        mat = np.hstack([random_x_pt, random_y_pt])
        PatchB_corners = PatchA_corners + mat
        H4Pt = (np.array(PatchA_corners - PatchB_corners))
        # H4Pt = H4Pt.flatten(order = 'F')

        H = cv2.getPerspectiveTransform(np.float32(PatchA_corners), np.float32(PatchB_corners))
        Hinv = np.linalg.inv(H)
        points2  = cv2.perspectiveTransform(np.float32(points1), Hinv)
        [xmin, ymin] = np.int32(points2.min(axis=0).ravel())
        [xmax, ymax] = np.int32(points2.max(axis=0).ravel())
        trans = [-xmin,-ymin]
        Ht = np.array([[1,0,trans[0]],[0,1,trans[1]],[0,0,1]]) # translate
        warped_img = cv2.warpPerspective(img, Ht.dot(Hinv), (xmax-xmin, ymax-ymin),flags = cv2.INTER_LINEAR)
        PatchB =  warped_img[random_x + trans[1] : random_x + patch_size + trans[1], random_y + trans[0] : random_y + patch_size + trans[0]]
        print("PatchA, PatchB shape :",PatchA.shape,PatchB.shape)
        return warped_img, PatchA_corners, PatchB_corners, PatchA, PatchB, H4Pt

# def GenerateBatch( TImagesPath1,TImagesPath2, TLabelPath, ImageSize, MiniBatchSize):
#     """
#     Inputs:
#     BasePath - Path to COCO folder without "/" at the end
#     DirNamesTrain - Variable with Subfolder paths to train files
#     NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
#     TrainCoordinates - Coordinatess corresponding to Train
#     NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
#     ImageSize - Size of the Image
#     MiniBatchSize is the size of the MiniBatch
#     Outputs:
#     I1Batch - Batch of images
#     CoordinatesBatch - Batch of coordinates
    
#     """
#     # print("INside /////////")
#     transforms=tf.Compose([tf.ToTensor(),
#                             tf.Normalize((0.5,0.5),(0.5,0.5), inplace = True)])
#     TI1Batch = []
#     TCoordinatesBatch = []
    
#     if os.path.exists(TImagesPath1): 
#         Timage1_list = os.listdir(TImagesPath1)
#         # print("Timage1_list : ",Timage1_list)
#     else:
#         raise Exception ("Directory Image1 doesn't exist")
#     if os.path.exists(TImagesPath2): 
#         Timage2_list = os.listdir(TImagesPath2)
#     else:
#         raise Exception ("Directory Image2 doesn't exist")

#     Timages1_path, Timages2_path = [], []
#     for i in range(len(Timage1_list)):
#         Timage1_path = TImagesPath1+'/'+Timage1_list[i]
#         Timage2_path = TImagesPath2+'/'+Timage2_list[i]
#         Timages1_path.append(Timage1_path)
#         Timages2_path.append(Timage2_path)
#         # print("Timage1_path",Timages1_path,'\n')/
#         # if i == 10:
#         #     break
#     # print(os.listdir('../Data/Train/Patch/PatchA/'))
#     # print(Timages1_path)
#     Timages1 = [cv2.imread(i, 0) for i in Timages1_path]
#     # Timages2 = images1.copy()
#     # print("Timages1 : ",Timages1)
#     Timages2 = [cv2.imread(i,0) for i in Timages2_path]
#     TImageset1= Timages1
#     TImageset2 = Timages2
   

#     ImageNum = 0
#     while ImageNum < MiniBatchSize:
#         # Generate random image
#         TIdx = random.randint(0, len(TImageset1)-1)
#         # VIdx = random.randint(0, len(VImageset1)-1)


#         TImg1 = TImageset1[TIdx]
#         TImg2 = TImageset2[TIdx]
#         # VImg1 = VImageset1[VIdx]
#         # VImg2 = VImageset2[VIdx]
        
        
#         TImg1 = np.expand_dims(TImg1, 2)
#         TImg2 = np.expand_dims(TImg2, 2)        
#         TImg = np.concatenate((TImg1, TImg2), axis = 2)        
#         TImg = transforms(TImg)
#         print("TImg shape :",TImg.shape)

#         # VImg1 = np.expand_dims(VImg1, 2)
#         # VImg2 = np.expand_dims(VImg2, 2)        
#         # VImg = np.concatenate((VImg1, VImg2), axis = 2)  
             
#         # VImg = transforms(VImg)
        
       
#         Tlabel = np.genfromtxt(TLabelPath + str(TIdx+1) + '.csv', delimiter=',')
#         TCoordinates = Tlabel
        
#         TImg =  np.float32(TImg)
#         TImg = torch.from_numpy(TImg)
#         TI1Batch.append(TImg)
#         TCoordinatesBatch.append(TCoordinates)

       
#         # Vlabel = np.genfromtxt(VLabelPath + str(VIdx+1) + '.csv', delimiter=',')
#         # VCoordinates = torch.from_numpy(Vlabel)
        
#         # VImg =  np.float32(VImg)
#         # VImg = torch.from_numpy(VImg)
       
#         # VI1Batch.append(VImg)
#         # VCoordinatesBatch.append(torch.tensor(VCoordinates, dtype=torch.float32))
#         ImageNum += 1
       

#     return torch.stack(TI1Batch).to(device), TCoordinatesBatch

def plot_corners(img, patch_b_corners, pred_patch_b_corners, t, filename):
    cv2.polylines(img, np.int32([patch_b_corners + t]), isClosed = True, color = (255,255,0), thickness = 2)
    cv2.polylines(img, np.int32([pred_patch_b_corners + t]), isClosed = True, color = (0,255,0), thickness = 2)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    cv2.imwrite(filename,img)

def TestOperation(IPATH,ModelPath, numFiles):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    print("sdbsuwd/////////////////////csbdc")
    model = model = Net(2,128,128)
    patchSize = 128
    pert = 32
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    transforms=tf.Compose([tf.ToTensor(), tf.Normalize((0.5,0.5),(0.5,0.5), inplace = True)])
    for i in range(1,numFiles+1):
        print("i :",i)
        # Read grayscale image from file 
        img = cv2.imread(IPATH+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (320, 240), interpolation = cv2.INTER_AREA)
        # Find Warp Image
        warped_img, PatchA_corners, PatchB_corners, PatchA, PatchB, H4Pt,t = createPatch(img, patchSize,pert) 
        img1 = PatchA
        img2 =  PatchB
        Img1 = np.expand_dims(img1, 2)
        Img2 = np.expand_dims(img2, 2) 
        transform=tf.Compose([tf.ToTensor()])
        print("Img1 ",Img1.shape)
        print("Img2 :",Img2.shape)
        Img = np.concatenate((Img1, Img2), axis = 2)
        Img = transforms(Img)
        Img = Img.unsqueeze(0)
        print("Image Shape :",Img.shape)
        model.eval()
        # H4PT predicted form network
        H4pt_predicted = model(Img)
        # H4pt_predicted = H4pt_predicted.squeeze(0)
        # H4pt_predicted = H4pt_predicted.unsqueeze(0)
        print("H4pt_predicted  Shape :",H4pt_predicted.shape)
        # Convert to numpy from tensor
        H4pt_predicted =H4pt_predicted.detach().numpy()
        # reshape the shape to 4,2 in order to have simialr shape of cornersA
        H4pt_predicted = np.reshape(H4pt_predicted, (4,2))
        # patch B corners predicted from H4PT predicted 
        PatchB_corners_Pred = PatchA_corners - H4pt_predicted

        print("Predicted Patch B Corners :",PatchB_corners_Pred)
        print(" Patch B Corners :",PatchB_corners)
        filename = "/home/usivaraman/CV/CV_P1/Phase2/Code/Results/Supervised/" + str(i) +"_out.png"
        plot_corners(warped_img, PatchB_corners, PatchB_corners_Pred, t, filename)






def test_sup():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ImagePath",
        dest="ImagePath",
        default="../Phase2/Code/Data/Test/GImg/",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--CheckPoint",
        dest="CheckPoint",
        default="../CheckPoints/Supervised/49model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    # print("Runninggggggggg")
    Args = Parser.parse_args()
    ImagePath = Args.ImagePath
    CheckPoint = Args.CheckPoint
    TestOperation( ImagePath,CheckPoint, 1000)
    # print("sdbsucsbdc")

# if __name__ == "__main__":
#     print("Runninf")
#     main()
