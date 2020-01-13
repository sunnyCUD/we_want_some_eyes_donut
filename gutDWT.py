#!/usr/bin/env python
# coding: utf-8

# In[68]:


import cv2
import glob
import numpy as np
from pywt import wavedec2
from matplotlib import pyplot as plt
import pywt
import pandas as pd
import scipy
from scipy import ndimage
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


# In[69]:


def dwtopticdiscfinder(img):
    height, width = img.shape[:2]
    img2= cv2.resize(img,(int(width/2),int(height/2)))
    b,g,r = cv2.split(img2)
    #coeffs0 =np.zeros((42, 33),dtype='float64')
    coeffs0 =np.zeros((33, 42),dtype='float64')
    coeffs = pywt.wavedec2(g, 'db2', level=1) 
    coeffs2 = pywt.wavedec2(g, 'db2', level=5) 
    #print(coeffs2[0].shape[:2])

    a0=coeffs2[1][0]
    b0=coeffs2[1][1]
    c0=coeffs2[1][2]
    a1=coeffs2[2][0]
    b1=coeffs2[2][1]
    c1=coeffs2[2][2]
    a2=coeffs2[3][0]
    b2=coeffs2[3][1]
    c2=coeffs2[3][2]
    a3=coeffs2[4][0]
    b3=coeffs2[4][1]
    c3=coeffs2[4][2]
    a4=coeffs2[5][0]
    b4=coeffs2[5][1]
    c4=coeffs2[5][2]
    coeffsN=[coeffs0,(a0,b0,c0),(a1,b1,c1),(a2,b2,c2),(a3,b3,c3),(a4,b4,c4)]
    re=pywt.waverec2(coeffsN, 'db2')
    #crate kernel for erosion dilation
    #print("before")
    #print(re.min())
    #print(re.max())
    if re.min() < 0:
        re = re + np.abs(re.min())
    if re.min() > 0:
        re = re - re.min()
    re = re*254/re.max()
    #print("after")
    #print(re.min())
    #print(re.max())
    #re=255-re
    #plt.imshow(re, cmap='gray')  # show it in grayscale
    #plt.show()  # display!
    H=g-re
    #H = np.abs(H)
    #plt.imshow(H, cmap='gray')  # show it in grayscale
    #plt.show()  # display!
    #plt.imshow(g, cmap='gray')  # show it in grayscale
    #plt.show()  # display!
    #H=x-re
    #plt.imshow(H, cmap='gray')  # show it in grayscale
    #plt.show()  # display!
    kernel1_size = 12
    kernel2_size = 60
    
    erosion = cv2.erode(H,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12)),iterations = 1) #เปลี่ยนgเป็นรูปที่มึงต้องการ
    dilation = cv2.dilate(erosion,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(60,60)),iterations = 1)
    #median
    M=scipy.ndimage.median_filter(dilation,1)
    #plt.imshow(M, cmap='gray')  # show it in grayscale
    #plt.show()  # display!
    #median
    #median = cv2.medianBlur(dilation, 5)
    #find max intesity
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(M)
    ret,thresh1 = cv2.threshold(dilation,int(maxVal)-1,255,cv2.THRESH_BINARY)
    #calculate center of mass
    M = cv2.moments(thresh1)
    #calculate x,y coordinate of center
    if M["m00"]:
        cX = float(M["m10"] / M["m00"])
        cY = float(M["m01"] / M["m00"])    
        #shift compensation
        if cX > 632:
            cX = cX+32
        else:
            cX = cX-32
        #print(cX)
        #print(cY)
        return cX*2,cY*2
    else:
        #print("center of mass not found")
        return None,None
    


# In[67]:


#example
'''
path = 'RAW_FUNDUS_INPUT/*.jpg'
image_list = []
for filename in glob.glob(path):
    image_list.append(filename)
    
name_list = []
x_list = []
y_list = []
for i in image_list:
    img = cv2.imread(i) #import image
    height, width = img.shape[:2]
    x,y = dwtopticdiscfinder(img)
    cv2.circle(img,(int(x),int(y)),5,(255,0,0),-1)
    img2= cv2.resize(img,(1264,968))
    save_path = "outputDWLT/"+i.replace(path.replace("/*.jpg","")+"\\","")
    name_list.append(i.replace(path.replace("/*.jpg","")+"\\",""))
    x_list.append(int(x))
    y_list.append(int(y))
    #print(save_path)
    cv2.imwrite(save_path,img2)
    
    #cv2.imshow("out",img2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
df = pd.DataFrame({'file name':name_list,
                   'x':x_list,
                   'y':y_list
                   })
writer = ExcelWriter('GUT.xlsx')
df.to_excel(writer,'Sheet1',index = False)
writer.save()
'''

