import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
import glob


xc = []
yc = []
xdu = []
ydu = []
xdl = []
ydl = []
xcu = []
ycu = []
xcl = []
ycl = []
name = []

def activecontour(original_image,ROI_image,x_ROI,y_ROI,ROI_size,radius,B,WL,WE):
    """
    get disc region label
    Parameters:
        -original_image: raw image 
        -ROI_image: image with regoin of interest
        -x_ROI: x coordinate at center of ROI image
        -y_ROI: y coordinate at center of ROI image
        - ROI_size: width and height of region of interest
        - imgname: name of raw image
    Returns:
        -name: list of name of raw image
        -xdu: list of x coordinate at the top of disc active contour region
        -ydu: list of y coordinate at the top of disc active contour region
        -xdl: list of x coordinate at the bottom of disc active contour region
        -ydl: list of y coordinate at the bottom of disc active contour region
    """
  
    """
        OpenCV channel configuration is BGR 
        Need to change to RGB for matplotlib
    """
    
    img = ROI_image
    original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    row, column = np.shape(original[:,:,0])
    xc = x_ROI
    yc = y_ROI
    
    
    """
        set matplotlib to show hsv color in figure
        change colorspace of ROI image from BGR to HSV colorspace
    """
    
    plt.hsv()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    """
        eliminate noise in image by fastNlMeansDenoising
    """
    
    img_denoise = cv2.fastNlMeansDenoising(hsv[:,:,2],None,7,21)
    
    
    """
        create initial contour for snake active contour
        create snake active contour from library active_contour from skimage.segmentation
    """
    s = np.linspace(0, 2*np.pi, 400)
    r = int(img.shape[0]/2)
    c = int(img.shape[1]/2)
    #radius = 300
    r = r + radius*np.sin(s) 
    c = c + radius*np.cos(s)
    init = np.array([r, c]).T
    
    snake = active_contour(img_denoise, init, alpha=0.0275, beta=B, w_line=WL, w_edge=WE, gamma=0.001
                           ,boundary_condition="periodic", max_px_move=1.0, max_iterations=2000, 
                           convergence=0.5, coordinates='rc')
    
    realsnake = snake
    realsnake[:,1] = xc-(ROI_size/2)+snake[:, 1]
    realsnake[:,0] = yc-(ROI_size/2)+snake[:, 0]
    thesnake = np.array([realsnake[:,1], realsnake[:,0]]).T
    
    disc_label = cv2.cvtColor(cv2.polylines(original, np.int32([thesnake]), 
                                            1, (0,0,255),thickness=2), cv2.COLOR_RGB2BGR)
    
    """
        find position of upper and lower of active contour disc region
    """
    
    maxindex = 0
    minindex = 0
    
    minindex = np.where(np.isin(snake[:,0], min(snake[:, 0])))
    maxindex = np.where(np.isin(snake[:,0], max(snake[:, 0])))
#        
#    yu = min(yc-(ROI_size/2)+snake[:, 0])
#    xu = yc-(ROI_size/2)+snake[minindex[0],1]
#    yl = max(yc-(ROI_size/2)+snake[:, 0])
#    xl = yc-(ROI_size/2)+snake[maxindex[0],1]
#    
    yu = min(thesnake[:, 0])
    xu = thesnake[minindex[0],1]
    yl = max(thesnake[:, 0])
    xl = thesnake[maxindex[0],1]
        
    return disc_label,xu,yu,xl,yl,snake
    
def disccuplabel(save_path):
    
    """
    Show disc label image from the save_path folder
    Parameters:
        - save_path: path of folder for image saving
    """
    
    directoryPath = "{}*.".format(save_path)
    types = ["jpg", "jpeg"]
    listofimages = []
    
    for extension in types:
        listofimages.extend(glob.glob( directoryPath + extension ))
    
    for image in listofimages:
        img = cv2.imread(image)
        cv2.namedWindow("Label",cv2.WINDOW_NORMAL)
        cv2.imshow("Label",img)
        cv2.resizeWindow("Label", 800, 600)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def SaveCoordinate(name,xcenter,ycenter,xdupper,ydupper,xdlower,ydlower): 
    
    """
    Save center coordinate, upper contour coordinate, lower contour coordinate
    Parameters:
        -name: list of name of raw image
        -xc: list of x coordinate at center of ROI image
        -yc: list of y coordinate at center of ROI image
        -xu: list of x coordinate at the top of active contour region
        -yu: list of y coordinate at the top of active contour region
        -xl: list of x coordinate at the bottom of active contour region
        -yl: list of y coordinate at the bottom of active contour region
    """
    
    workbook = xlsxwriter.Workbook('Disc Active Contour Edited.xlsx') 
    worksheet = workbook.add_worksheet() 
    
    worksheet.write('A1', 'File name') 
    worksheet.write('B1', 'X Center') 
    worksheet.write('C1', 'Y Center')
    worksheet.write('D1', 'X Disc Upper')
    worksheet.write('E1', 'Y Disc Upper')
    worksheet.write('F1', 'X Disc Lower')
    worksheet.write('G1', 'Y Disc Lower')
    
    rowexcel = 1
    for a in name:
        worksheet.write(rowexcel,0, a)
        rowexcel = rowexcel+1
    rowexcel = 1
    for j in xcenter:
        worksheet.write(rowexcel,1, j)
        rowexcel = rowexcel+1
    rowexcel = 1
    for k in ycenter:
        worksheet.write(rowexcel,2, k)
        rowexcel = rowexcel+1
    rowexcel = 1
    for j in xdupper:
        worksheet.write(rowexcel,3, j) 
        rowexcel = rowexcel+1
    rowexcel = 1
    for k in ydupper:
        worksheet.write(rowexcel,4, k)
        rowexcel = rowexcel+1
    rowexcel = 1
    for j in xdlower:
        worksheet.write(rowexcel,5, j) 
        rowexcel = rowexcel+1
    rowexcel = 1
    for k in ydlower:
        worksheet.write(rowexcel,6, k)
        rowexcel = rowexcel+1   
    
    workbook.close()

  
#import ROIfromexcel as ROI
#import cv2
#import pandas as pd
#import ActiveContourDisc as ad
#
##=========USER START================
##folder path
#img_folder_path = 'C:/Users/USER/Desktop/Validation/True/True/*.jpg'
#ROI_path = r'C:\Users\USER\Desktop\Validation\True\True\ROI.xlsx'
#ROI_size = 500
#save_folder_path = 'C:/Users/USER/Desktop/Validation/True/Evaluation/'
##=========USER END================
#
##create dataframe of ROI.xlsx
#df = pd.read_excel(ROI_path, index_col=0)
##crate list of image path in the folder
#img_path_list = ROI.readImgFolder(img_folder_path)
#list_name = []
#x_center = []
#y_center = []
#x_disc_upper = []
#y_disc_upper = []
#x_disc_lower = []
#y_disc_lower = []
#x_cup_upper = []
#y_cup_upper = []
#x_cup_lower = []
#y_cup_lower = []
#
#
#for img_path in img_path_list:
#    #read only name
#    img_name = ROI.getImName(img_path,img_folder_path)
#    #read x,y coordinate from ROI.xlsx
#    x_ROI,y_ROI = ROI.getCenterROI(img_name,df)
#    #read image
#    img = cv2.imread(img_path)
#    #get ROI image, x1, y1, x2, and y2
#    img_ROI,x1_ROI,y1_ROI,x2_ROI,y2_ROI = ROI.getROI(img,x_ROI,y_ROI,ROI_size)
#
#    #get disc label image from ROI image
#    name,xd_up, yd_up, xd_low, yd_low, xc_up, yc_up, xc_low, yc_low = ad.activecontour(img,img_ROI,x_ROI,y_ROI,ROI_size,img_name,save_folder_path)
#    
#    list_name.append(name)
#    x_center.append(x_ROI)
#    y_center.append(y_ROI)
#    x_disc_upper.append(xd_up)
#    y_disc_upper.append(yd_up)
#    x_disc_lower.append(xd_low)
#    y_disc_lower.append(yd_low)
#    x_cup_upper.append(xc_up)
#    y_cup_upper.append(yc_up)
#    x_cup_lower.append(xc_low)
#    y_cup_lower.append(yc_low)
#    
#ad.SaveCoordinate(list_name,x_center,y_center,x_disc_upper,y_disc_upper,x_disc_lower,y_disc_lower,
#                  x_cup_upper,y_cup_upper,x_cup_lower,y_cup_lower)
#    
#ad.disccuplabel(save_folder_path)
#
##=====================PUT YOUR CODE HERE===========================
##                    
##==================================================================