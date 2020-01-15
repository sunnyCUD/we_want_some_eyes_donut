#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import glob
import cv2
from ActiveContourDisc import activecontour
from ROI_from_excel import getROI
from ROI_from_excel import ShowResizedIm
from gutDWT import dwtopticdiscfinder
from cupB_eval import find_cup
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


# In[5]:


def getCDR(img,ROI_size):
    canvas = img.copy()
    ODC_flag = 0
    OD_flag = 0
    OC_flag = 0
    #1 img -> preprocess
    #2 proprocess -> coordinate
    xODC,yODC = dwtopticdiscfinder(img)
    xODC = int(xODC)
    yODC = int(yODC)
    if type(xODC) is not type(None) and type(yODC) is not type(None):
        ODC_flag = 1
    #3 coordinate -> ROI
    if ODC_flag:
        ROI_image,x_ROI1,y_ROI1,x_ROI2,y_ROI2 = getROI(img,xODC,yODC,ROI_size)
    #4 disc
    if ODC_flag:
        canvas,__,top_OD,__,bot_OD,snake = activecontour(canvas,ROI_image,xODC,
                                                   yODC,ROI_size,radius=250,B=90,WL=0.123,WE=6.5)
    if type(top_OD) is not type(None) and type(bot_OD) is not type(None) and type(snake) is not type(None):
        OD_flag = 1
        OD_size = round(bot_OD-top_OD,0)
    #5 cup
    if ODC_flag and OD_flag:
        ROI_imageL,x_ROI1,y_ROI1,x_ROI2,y_ROI2 = getROI(canvas,xODC,yODC,ROI_size)
        center, radious, area, error, image_result = find_cup(ROI_image,ROI_imageL)
    if type(center) is not type(None) and type(radious) is not type(None):
        OC_flag = 1
    if OC_flag:
        cv2.circle(canvas,(int(center[0]+x_ROI1),int(center[1]+y_ROI1)), radious,(0,255,0),2)
        OC_size = round(radious*2,0)
    #6 CDR
    if ODC_flag:
        cv2.line(canvas, (xODC+20, yODC), (xODC-20, yODC), (0, 0, 255), 3) 
        cv2.line(canvas, (xODC, yODC+20), (xODC, yODC-20), (0, 0, 255), 3) 
    if OD_flag and OC_flag:
        CDR = round(OC_size/OD_size,2)
        return canvas,OD_size,OC_size,CDR
    else:
        return canvas,None,None,None

# In[7]:


def main():
    #=============USER INPUT=============
    
    crop_size = 500
    
    #=============USER END===============
    path = 'RAW_FUNDUS_INPUT/*.jpg'
    image_list = []
    for filename in glob.glob(path):
        image_list.append(filename)
    name_list = []
    OD_size_list = []
    OC_size_list = []
    CDR_list = []
    for i in image_list:
        img = cv2.imread(i) #import image
        output,OD_size,OC_size,CDR = getCDR(img,crop_size)
        name_list.append(i)
        OD_size_list.append(OD_size)
        OC_size_list.append(OC_size)
        CDR_list.append(CDR)
        #ShowResizedIm(output,"image_result",2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    df = pd.DataFrame({'file name':name_list,
                           'OD_size':OD_size_list,
                           'OC_size':OC_size_list,
                           'CDR':CDR_list
                          })
    writer = ExcelWriter('CDR.xlsx')
    df.to_excel(writer,'Sheet1',index=False)
    writer.save()
if __name__ == '__main__':
    main()

