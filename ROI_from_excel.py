#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import xlrd 
import glob
import cv2

def getImName(image_path,folder_path):
    """
    get image name by clean up image path
    Parameters:
        -image_path: path of image in the folder
        -folder_path: path of folder that contain images
    Returns:
        -name: name of the image
    """
    name = image_path.replace(folder_path.replace("/*.jpg","")+"\\","")
    print(name)
    return name

def getCenterROI(image_name,df):
    """
    get center ROI from image name and dataframe
    Parameters:
        -image_name: name of image that match with name in dataframe
        -df: dataframe that contain image name and ROI center coordinate
    Returns:
        -x,y: coordinate of ROI center
    """
    x = df.loc[image_name][0]
    y = df.loc[image_name][1]
    return x,y

def getROI(img,x,y,ROI_size):
    """
    crop square ROI from image with center x, y coordinate
    Parameters:
        -img: input image
        -x,y: center coordinate of ROI
        -ROI_size: size of ROI in pixel
    Returns:
        -imCrop: image of ROI
        -x1,y1: coordinate of top left corner of the ROI
        -x2,y2: coordinate of botton right corner of the ROI
    """
    x1 = int(x-(ROI_size/2))
    y1 = int(y-(ROI_size/2))
    x2 = int(x+(ROI_size/2))
    y2 = int(y+(ROI_size/2))
    imCrop = img[y1:y2,x1:x2]
    return imCrop,x1,y1,x2,y2

def readImgFolder(path):
    """
    read image form folder and create a list of images path in side the folder
    Parameters:
        -path: folder path (ex.'RAW_FUNDUS_INPUT/*.jpg')
    Returns:
        -image_list: list of images path in the folder
    """
    image_list = []
    for filename in glob.glob(path):
        image_list.append(filename)
    return image_list

def ShowResizedIm(img,windowname,scale):
    """
    opencv imshow resized image on a new window
    Parameters:
        -img: image
        -window: window name
        -scale: size of the display image will be divided by this value(ex. scale=2 will make image 2 time smaller)
    """
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    height, width = img.shape[:2]   #get image dimension
    cv2.resizeWindow(windowname,int(width/scale) ,int(height/scale))                    # Resize image
    cv2.imshow(windowname, img)                            # Show image
'''
example
#=========USER START================
#folder path
img_folder_path = 'RAW_FUNDUS_INPUT/*.jpg'
ROI_path = 'ROI.xlsx'
ROI_size = 500
#=========USER END================

#create dataframe of ROI.xlsx
df = pd.read_excel(ROI_path, index_col=0)
#crate list of image path in the folder
img_path_list = readImgFolder(img_folder_path)
for img_path in img_path_list:
    #read only name
    img_name = getImName(img_path,img_folder_path)
    #read x,y coordinate from ROI.xlsx
    x_ROI,y_ROI = getCenterROI(img_name,df)
    #read image
    img = cv2.imread(img_path)
    #get ROI image, x1, y1, x2, and y2
    img_ROI,x1_ROI,y1_ROI,x2_ROI,y2_ROI = getROI(img,x_ROI,y_ROI,ROI_size)
    #show imCrop
    ShowResizedIm(img_ROI,"image ROI",2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #=====================PUT YOUR CODE HERE===========================
    #                    
    #==================================================================
'''

