#!/usr/bin/env python
# coding: utf-8

# In[22]:


import cv2
import glob
import numpy as np

def findClickCoordinate(img,dot,dot_size = 5):
    """
    Find coordinate of clicked location
    Parameters:
        -img: input image
        -dot: amount of output dot
        -dot_size: size of dot
    Returns:
        -circles: list of clicked coordinate [(x,y),...]
        -img: output image
    """
    def mouse_drawing(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print("Left click")
            circles.append((x, y))
    #create window
    cv2.namedWindow("Frame")
    #set mouse call back
    cv2.setMouseCallback("Frame", mouse_drawing)
    #create lit to contain coordinate
    circles = []
    while True:
        for center_position in circles:
            cv2.circle(img, center_position, dot_size, (0, 0, 255), -1)

        cv2.imshow("Frame", img)
        if len(circles) == dot:
            break
        key = cv2.waitKey(30)
        if key == 27:
            print("esc")
            break
        elif key == ord("d"):
            circles = []
    cv2.destroyAllWindows()#test
    return circles,img


# In[24]:


#=========USER START================
#folder path
path = 'RAW_FUNDUS_INPUT/*.jpg' 
image_number = 2
#=========USER END================

#read image
image_list = []
for filename in glob.glob(path):
    image_list.append(filename)
img = cv2.imread(image_list[image_number])
#find clicked coordinate
coor,out= findClickCoordinate(img,1,dot_size = 5)
#print coordinate
print(coor)
#show image
cv2.imshow("Output", out)
cv2.waitKey(0)#test
cv2.destroyAllWindows()#test

