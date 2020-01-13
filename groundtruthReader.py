#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import glob
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import time
import progressbar
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


def ShowResizedIm(img,windowname,scale):
   cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
   height, width = img.shape[:2]   #get image dimension
   cv2.resizeWindow(windowname,int(width/scale) ,int(height/scale))                    # Resize image
   cv2.imshow(windowname, img)                            # Show image


# In[5]:


#=========USER START================
#folder path
path = 'True labeled/*.jpg' 
#toggle display 0 or 1
display = 1
#toggle file save 0 or 1
save_excel = 0
#write
write_image = 0
#CDR threshold value
CDR_thresh = 0.7
#=========USER END================


# In[6]:


image_list = []
for filename in glob.glob(path):
    image_list.append(filename)

name_list = []
cup_size_list = []
disc_size_list = []
ratio_list = []
diag_result_list = []
# define the list of boundaries
boundaries = [
    ([5, 5, 5], [255, 20, 20]),    #blue
    ([5, 5, 5], [20, 255, 20])     #green
]
font = cv2.FONT_HERSHEY_SIMPLEX
with progressbar.ProgressBar(max_value=len(image_list)) as bar:
    progress = 0
    bar.update(progress)
    for img_source in  image_list:
        img = cv2.imread(img_source)
        height, width = img.shape[:2]
        name_list.append(str(img_source))
        # create NumPy arrays from the boundaries
        lower = np.array( boundaries[0][0], dtype = "uint8")
        upper = np.array( boundaries[0][1], dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(img, lower, upper)
        outputB = cv2.bitwise_and(img, img, mask = mask)

        # create NumPy arrays from the boundaries
        lower = np.array( boundaries[1][0], dtype = "uint8")
        upper = np.array( boundaries[1][1], dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(img, lower, upper)
        outputG = cv2.bitwise_and(img, img, mask = mask)
        BGray,__,__ = cv2.split(outputB)
        __,GGray,__ = cv2.split(outputG)
        ret1,threshB = cv2.threshold(BGray,100,255,cv2.THRESH_BINARY)
        ret2,threshG = cv2.threshold(GGray,100,255,cv2.THRESH_BINARY)

        canvas = img.copy()
        disc_size = 0
        cup_size = 0
        if ret1:
            __,contoursB, hierarchyB = cv2.findContours(threshB, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            topmost_list = []
            bottommost_list = []
            for cnt in contoursB:
                topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
                bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
                topmost_list.append(topmost[1])
                bottommost_list.append(bottommost[1])

            if len(topmost_list) and len(bottommost_list):
                cup_size = max(bottommost_list)-min(topmost_list)
                cup_size_list.append(cup_size)
                if display or write_image:
                    cv2.line(canvas,(0,min(topmost_list)),(width,min(topmost_list)),(255,255,0),2)
                    cv2.line(canvas,(0,max(bottommost_list)),(width,max(bottommost_list)),(255,255,0),2)
            else:
                cup_size_list.append("none")

        if ret2:
            __,contoursG, hierarchyG = cv2.findContours(threshG, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            topmost_list = []
            bottommost_list = []
            for cnt in contoursG:
                topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
                bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
                topmost_list.append(topmost[1])
                bottommost_list.append(bottommost[1])

            if len(topmost_list) and len(bottommost_list):
                disc_size = max(bottommost_list)-min(topmost_list)
                disc_size_list.append(disc_size)
                if display or write_image:
                    cv2.line(canvas,(0,min(topmost_list)),(width,min(topmost_list)),(0,255,255),2)
                    cv2.line(canvas,(0,max(bottommost_list)),(width,max(bottommost_list)),(0,255,255),2)
            else:
                disc_size_list.append("none")

        if cup_size and disc_size:
            ratio = round(cup_size/disc_size,2)
            ratio_list.append(ratio)
            if ratio>CDR_thresh:
                diag_result_list.append("Positive")
            else:
                diag_result_list.append("Nagative")
            if display or write_image:
                cv2.putText(canvas,"CDR: "+str(ratio),(10,200), font, 2.5,(255,255,255),5,cv2.LINE_AA)
        else:
            ratio_list.append("none")
            diag_result_list.append("none")
            if display or write_image:
                cv2.putText(canvas,"error ratio not found",(10,200), font, 2.5,(0,0,255),5,cv2.LINE_AA)
        if display:
            cv2.putText(canvas,str(img_source),(10,100), font, 2.5,(255,255,255),5,cv2.LINE_AA)
            ShowResizedIm(np.hstack([canvas,]),"mark",2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if write_image:
            cv2.putText(canvas,str(img_source),(10,100), font, 2.5,(255,255,255),5,cv2.LINE_AA)
            cv2.imwrite("output/"+str(progress)+".jpg", canvas) 
        bar.update(progress)
        progress = progress+1
if save_excel:    
    df = pd.DataFrame({'file name':name_list,
                       'disc size':disc_size_list,
                       'cup size':cup_size_list,
                       'ratio':ratio_list,
                       'result':diag_result_list
                      })
    writer = ExcelWriter('ActiveCon CDR.xlsx')
    df.to_excel(writer,'Sheet1',index=False)
    writer.save()


# In[ ]:


cv2.destroyAllWindows()

