#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
import numpy as np

def getROI(img):
    cv2.circle(img , (100,100), 10, (255, 255, 255), -1)
    ROI = 2.1
    return img,ROI

