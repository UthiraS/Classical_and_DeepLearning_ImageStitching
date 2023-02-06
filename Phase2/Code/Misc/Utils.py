#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
import random
# Add any python libraries here
import os
import zipfile
import argparse
import matplotlib.pyplot as plt
import sys
"""
Generating Patches
"""
def createPatch(img,patchSize = 128,pert= 32):
    boundary = 32

    img = cv2.resize(img, (320, 240), interpolation = cv2.INTER_AREA) 
            # if(img.shape[0]>240 and img.shape[1] >240):
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        




def dataGenerator(dataType,patchSize,pert):

    mainPath = "Data/"
    mPath= "Data"
    labelPath = "TxtFiles/"   

   
    if(dataType=="Train"):    
        ZipFileStrings= ["Train","Val"]
        numFilesList =[5000,1000]
        for i,ZipFileString in enumerate(ZipFileStrings):  
            
            IPATH = mainPath + ZipFileString  +'/Img/'
            PPATH = mainPath + ZipFileString  + '/Patch/' 
            ZString = ZipFileString+".zip"
            ZPATH = os.path.join(mPath , ZString)
            PatchAPath= PPATH +"/PatchA/"
            PatchBPath= PPATH +"/PatchB/"
            PatchesPath = PPATH +"/PATCHES/"
            CornersAPath = PPATH +"/CORNERS/"
            # print("IPATH :",IPATH)
            # print("PPATH :",PPATH)
            numFiles = numFilesList[i]
            print("ZipfileStringPath :",ZPATH)
            i =1
            count = 1
            while(i<numFiles+1):
                print("I :",i)
                # if i != 4:
                #     continue
                with zipfile.ZipFile(ZPATH, 'r') as zfile:
                    data        = zfile.read(ZipFileString+'/'+str(i)+'.jpg')
                    img         = cv2.imdecode(np.frombuffer(data, np.uint8), 1)   
                    img = cv2.resize(img, (320, 240), interpolation = cv2.INTER_AREA) 
                    # if(img.shape[0]>240 and img.shape[1] >240):
                    img_gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(IPATH+str(count)+'.jpg', img_gray)
                    count = count +1
                    i = i+ 1
                    # else:
                    #     i = i +1 
                    #     continue
                    
            
            for i in range(1,count):
                print("i :",i)
                warped_img, PatchA_corners, PatchB_corners, PatchA, PatchB, H4Pt = createPatch(cv2.imread(IPATH+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE), patchSize,pert) 
                patchA_img =PatchAPath +str(i)+".png"
                cv2.imwrite(patchA_img,PatchA)
                patchB_img = PatchBPath+str(i)+".png"
                cv2.imwrite(patchB_img,PatchB)             
                H4Pt = H4Pt.flatten(order = 'F')
                h4_filename = labelPath + "/"+ ZipFileString+ "/" + str(i) + ".csv"
                np.savetxt(h4_filename, H4Pt, delimiter = ",")

                PatchA_corners = PatchA_corners.flatten(order = 'F')
                Cornerfilename = CornersAPath + str(i) + ".csv"

                np.savetxt(Cornerfilename, PatchA_corners, delimiter = ",")
        
    elif(dataType=="Test"):    
        ZipFileString= "Test"
        
        numFiles = 1000
    
        ImagePATH = mainPath + ZipFileString  +'/Img/'
        IPATH = mainPath + ZipFileString  +'/GImg/'
        PPATH = mainPath + ZipFileString  + '/Patch/' 
        
        PatchAPath= PPATH +"/PatchA/"
        PatchBPath= PPATH +"/PatchB/"
        PatchesPath = PPATH +"/PATCHES/"
        CornersAPath = PPATH +"/CORNERS/"
        
        i =1
        
        while(i<numFiles+1):
            
            img = cv2.imread(ImagePATH+ str(i) + '.jpg',1)
            img = cv2.resize(img, (320, 240), interpolation = cv2.INTER_AREA) 
            # if(img.shape[0]>240 and img.shape[1] >240):
            img_gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(IPATH+str(i)+'.jpg', img_gray)
            
            i = i+ 1
                
                
        
        for i in range(1,numFiles):
            print("i :",i)
            warped_img, PatchA_corners, PatchB_corners, PatchA, PatchB, H4Pt = createPatch(cv2.imread(IPATH+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE), patchSize,pert) 
            patchA_img =PatchAPath +str(i)+".png"
            cv2.imwrite(patchA_img,PatchA)
            patchB_img = PatchBPath+str(i)+".png"
            cv2.imwrite(patchB_img,PatchB)             
            H4Pt = H4Pt.flatten(order = 'F')
            h4_filename = labelPath + ZipFileString+ "/" + str(i) + ".csv"
            np.savetxt(h4_filename, H4Pt, delimiter = ",")

            PatchA_corners = PatchA_corners.flatten(order = 'F')
            Cornerfilename = CornersAPath + str(i) + ".csv"

            np.savetxt(Cornerfilename, PatchA_corners, delimiter = ",")



# def main():
#     # Add any Command Line arguments here
#     Parser = argparse.ArgumentParser()
#     Parser.add_argument('--NumFeatures', default=128, help='Number of best features to extract from each image, Default:100')
#     Parser.add_argument('--Data', default='Train', help='The data that need to be prepared \'Train\' or \'Test\', Default:\'Train\'')
#     Args = Parser.parse_args()
#     NumFeatures = Args.NumFeatures
#     data_type   = Args.Data

#     patchSize =  NumFeatures 
#     pertubations= 32

#     dataGenerator(data_type,patchSize,pertubations)
    



   


# if __name__ == "__main__":
#     main()
    # print("running")
