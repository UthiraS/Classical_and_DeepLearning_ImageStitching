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
from Misc import Utils
import Train
import Train_UnSup
import Test
# Add any python libraries here
import os
import zipfile
import argparse
import matplotlib.pyplot as plt
import sys

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=128, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--Data', default='Train', help='The data that need to be prepared \'Train\' or \'Test\', Default:\'Train\'')
    Parser.add_argument('--ModelType', default='Sup', help='The data that need to be prepared \'Train\' or \'Test\', Default:\'Train\'')
    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    data_type   = Args.Data
    ModelType   = Args.ModelType

    patchSize =  NumFeatures 
    pertubations= 32

    Utils.dataGenerator(data_type,patchSize,pertubations)

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
    if(data_type== 'Train'):

        if(ModelType=='Sup'):
            Train.train_sup()
        elif(ModelType=='UnSup'):
            Train_UnSup.train_unsup()
    elif(data_type== 'Test'):
        if(ModelType=='Sup'):
            Test.test_sup()



    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
