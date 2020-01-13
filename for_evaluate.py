import numpy as np
import cv2
import math
import glob
from function import ShowResizedIm, FrangiFilter2D, rotate, getSkeletonIntersection, skeleton_endpoints
from skimage.morphology import skeletonize
from skimage import util 
import statistics

def ShowResizedIm(img,windowname,scale):
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    height, width = img.shape[:2]   #get image dimension
    cv2.resizeWindow(windowname,int(width/scale) ,int(height/scale))                    # Resize image
    cv2.imshow(windowname, img)                      
    
def Hessian2D(I,Sigma):

    if Sigma<1:
        print("error: Sigma<1")
        return -1
    I=np.array(I,dtype=float)
    Sigma=np.array(Sigma,dtype=float)
    S_round=np.round(3*Sigma)

    [X,Y]= np.mgrid[-S_round:S_round+1,-S_round:S_round+1]

    DGaussxx = 1/(2*math.pi*pow(Sigma,4)) * (X**2/pow(Sigma,2) - 1) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
    DGaussxy = 1/(2*math.pi*pow(Sigma,6)) * (X*Y) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))   
    DGaussyy = 1/(2*math.pi*pow(Sigma,4)) * (Y**2/pow(Sigma,2) - 1) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
  
    Dxx = signal.convolve2d(I,DGaussxx,boundary='fill',mode='same',fillvalue=0)
    Dxy = signal.convolve2d(I,DGaussxy,boundary='fill',mode='same',fillvalue=0)
    Dyy = signal.convolve2d(I,DGaussyy,boundary='fill',mode='same',fillvalue=0)

    return Dxx,Dxy,Dyy

def eig2image(Dxx,Dxy,Dyy):
    Dxx=np.array(Dxx,dtype=float)
    Dyy=np.array(Dyy,dtype=float)
    Dxy=np.array(Dxy,dtype=float)

    tmp = np.sqrt( (Dxx - Dyy)**2 + 4*Dxy**2)

    v2x = 2*Dxy
    v2y = Dyy - Dxx + tmp

    mag = np.sqrt(v2x**2 + v2y**2)
    i=np.array(mag!=0)

    v2x[i==True] = v2x[i==True]/mag[i==True]
    v2y[i==True] = v2y[i==True]/mag[i==True]

    v1x = -v2y 
    v1y = v2x

    mu1 = 0.5*(Dxx + Dyy + tmp)
    mu2 = 0.5*(Dxx + Dyy - tmp)

    check=abs(mu1)>abs(mu2)
            
    Lambda1=mu1.copy()
    Lambda1[check==True] = mu2[check==True]
    Lambda2=mu2
    Lambda2[check==True] = mu1[check==True]
    
    Ix=v1x
    Ix[check==True] = v2x[check==True]
    Iy=v1y
    Iy[check==True] = v2y[check==True]
    
    return Lambda1,Lambda2,Ix,Iy

def FrangiFilter2D(I):
    I=np.array(I,dtype=float)
    defaultoptions = {'FrangiScaleRange':(1,10), 'FrangiScaleRatio':2, 'FrangiBetaOne':0.5, 'FrangiBetaTwo':15, 'verbose':True,'BlackWhite':True};  
    options=defaultoptions

    sigmas=np.arange(options['FrangiScaleRange'][0],options['FrangiScaleRange'][1],options['FrangiScaleRatio'])
    sigmas.sort()

    beta  = 2*pow(options['FrangiBetaOne'],2)  
    c     = 2*pow(options['FrangiBetaTwo'],2)

    shape=(I.shape[0],I.shape[1],len(sigmas))
    ALLfiltered=np.zeros(shape) 
    ALLangles  =np.zeros(shape) 

    Rb=0
    S2=0
#    for i in range(len(sigmas)):
#       if(options['verbose']):
#           print('Current Frangi Filter Sigma: ',sigmas[i])
        
    [Dxx,Dxy,Dyy] = Hessian2D(I,sigmas[2])

    Dxx = pow(sigmas[2],2)*Dxx  
    Dxy = pow(sigmas[2],2)*Dxy  
    Dyy = pow(sigmas[2],2)*Dyy
         
        #Calculate (abs sorted) eigenvalues and vectors  
    [Lambda2,Lambda1,Ix,Iy]=eig2image(Dxx,Dxy,Dyy)  

        #Compute the direction of the minor eigenvector  
    angles = np.arctan2(Ix,Iy)  

        #Compute some similarity measures  
    Lambda1[Lambda1==0] = np.spacing(1)

    Rb = (Lambda2/Lambda1)**2  
    S2 = Lambda1**2 + Lambda2**2
        
        #Compute the output image
    Ifiltered = np.exp(-Rb/beta) * (np.ones(I.shape)-np.exp(-S2/c))
         
    if(options['BlackWhite']): 
        Ifiltered[Lambda1<0]=0
    else:
        Ifiltered[Lambda1>0]=0
        
        #store the results in 3D matrices  
    ALLfiltered[:,:,2] = Ifiltered 
    ALLangles[:,:,2] = angles

        # Return for every pixel the value of the scale(sigma) with the maximum   
        # output pixel value  
    if len(sigmas) > 1:
        outIm=ALLfiltered.max(2)
    else:
        outIm = (outIm.transpose()).reshape(I.shape)
            
    return outIm

def rotate(origin, xy, radians):
    """Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    y, x = xy[:2]
    offset_y, offset_x = origin[:2]
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    
    
    return qx, qy

def neighbour(x,y,image):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    img = image.copy()
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]   

def getSkeletonIntersection(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.
    
    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of
    
    Returns: 
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A big list of valid intersections                  2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6 
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]];
    image = skeleton.copy();
    image = image/255;
    row,col = image.shape[:2]
    intersections = []
    neighbours = []
    for x in range(1,row-1):
        for y in range(1,col-1):
            # If we have a white pixel
            if image[x,y] == 1:
                neighbours = neighbour(x,y,image)
                valid = True;
                if neighbours in validIntersection:
                    intersections.append((y,x));
                    
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections;

def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    out = np.zeros_like(skel)
    out = np.where(filtered==11)
    return out

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def extract_bv(image):
    b,green_fundus,r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, 
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(fundus_eroded.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    
    return blood_vessels

def find_cup(img,img_disc):
    blue,green,red = cv2.split(img_disc)
    row = height = img.shape[0]
    col = width = img.shape[1]
    
    ########### Disc boundary ###############
    hsv = cv2.cvtColor(img_disc, cv2.COLOR_BGR2HSV)
    ##### Blue
    lower_range = np.array([110,50,50])
    upper_range = np.array([130, 255,255])
    
    disc = cv2.inRange(hsv, lower_range, upper_range)

    kernel = np.ones((5,5),np.uint8)
    disc = cv2.dilate(disc, kernel, iterations = 1)
    
    image_result = img.copy()
    contours, _ = cv2.findContours(disc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        hull= cv2.convexHull(contours[0])
        cv2.ellipse(disc, cv2.fitEllipse(hull), (255,255,255), thickness=cv2.FILLED)
        cv2.ellipse(image_result, cv2.fitEllipse(hull), (0,255,0), 2)

    ########### SAMPLING IMAGE ###############
    sampling = np.zeros((row, col))
    center = [int(row/2),int(col/2)]
    #cv2.circle(sampling, (center[0],center[1]), 1, (255, 255, 255), 1)
    i=0
    mum = 4
    
    while(i<center[1]-40):
        sampling_posO = [center[0],center[1]+i+40]
    #    sampling[sampling_pos[0],sampling_pos[1]] = 255
        for j in range(0,360,mum):
            theta = np.radians(j)
            x,y = rotate(center, sampling_posO, theta)
            sampling_pos = [round(y),round(x)]
            sampling[sampling_pos[0],sampling_pos[1]] = 255
    #        cv2.circle(sampling, (sampling_pos[0], sampling_pos[1]), 1, (255, 255, 255), 1)
        i+=5
        
    ##substract sampling by disc
    kernel = np.ones((15,15),np.uint8)
    disc = cv2.erode(disc,kernel,iterations = 1)
    disc= disc.astype('float64')
    heuristic = cv2.bitwise_and(sampling, disc)
    
    ########### vessel curvature detection ###############
    blood = cv2.normalize(green.astype('double'), None, 0.0, 1, cv2.NORM_MINMAX) # Convert to normalized floating point
    outIm=FrangiFilter2D(blood)
    img_frangi=outIm*(1000000)
    _,thresh = cv2.threshold(img_frangi,1.0,1,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    
    skeleton = skeletonize(thresh)
    skeleton = skeleton*255
    skeleton = skeleton.astype(np.uint8)
    skeleton_copy = skeleton.copy()
    disc = disc.astype(np.uint8)
    skeleton = cv2.bitwise_and(skeleton, disc)
    
    #store skeelton's white pixels
    points = []
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i,j]==255:
                points.append([j,i])
    
    intersections = getSkeletonIntersection(skeleton)
    for i in range(len(intersections)):
        cv2.circle(skeleton_copy, intersections[i], 2, (255,255,255), 2) 
        
    end_pointTU = skeleton_endpoints(skeleton)
    end_points = []
    for i in range(len(end_pointTU[0])):
        cv2.circle(skeleton_copy, (end_pointTU[1][i],end_pointTU[0][i]), 2, (255,255,255), 2)
        x = end_pointTU[1][i]
        y = end_pointTU[0][i]
        end_points.append([x,y])
    
    ##store all intersect points to end_point in list [x,y]
    #double the numbers inside
    for i in range(len(intersections)):   
        end_points.append(list(intersections[i]))
        
    ###check and divide each line with all coordinate
    line = [[]]
    counts = 0
    while points!=[]:
        if end_points != []:
            x,y = end_points[0][:2]
        else:
            skeletonTemp = np.zeros((skeleton.shape[0],skeleton.shape[1]))
            for i in range(len(points)):
                skeletonTemp[points[i][1],points[i][0]] = 255
            end_pointTU = skeleton_endpoints(skeletonTemp)
            a=True
            for i in range(len(end_pointTU)):
                if len(end_pointTU[i])!=0:
                    a and False
                else:
                    a and True
            if a==True:
                points=[]
                break
            end_points = []
            for i in range(len(end_pointTU[0])):
                x = end_pointTU[1][i]
                y = end_pointTU[0][i]
                end_points.append([x,y])
            x,y = end_points[0][:2]
        end_points.remove([x,y])
        points.remove([x,y])
        check = 0
        line.append([])
        line[counts].append([x,y])
        while check!=1:                
            if skeleton[y,x+1]==255 and [x+1,y] in points:
                x = x+1
                points.remove([x,y])
                line[counts].append([x,y])
            elif skeleton[y,x-1]==255 and [x-1,y] in points:
                x = x-1
                points.remove([x,y])
                line[counts].append([x,y])
            elif skeleton[y+1,x]==255 and [x,y+1] in points:
                y=y+1
                points.remove([x,y])
                line[counts].append([x,y])
            elif skeleton[y-1,x]==255 and [x,y-1] in points:
                y=y-1
                points.remove([x,y])
                line[counts].append([x,y])
            elif skeleton[y+1,x+1]==255 and [x+1,y+1] in points:
                y=y+1
                x=x+1
                points.remove([x,y])
                line[counts].append([x,y])
            elif skeleton[y+1,x-1]==255 and [x-1,y+1] in points:
                y=y+1
                x=x-1
                points.remove([x,y])
                line[counts].append([x,y])
            elif skeleton[y-1,x+1]==255 and [x+1,y-1] in points:
                y=y-1
                x=x+1
                points.remove([x,y])
                line[counts].append([x,y])
            elif skeleton[y-1,x-1]==255 and [x-1,y-1] in end_points:
                y=y-1
                x=x-1
                points.remove([x,y])
                line[counts].append([x,y])
            else:
                if [x,y] not in end_points:
                        check=1       
            if [x,y] in end_points:
                check=1
                end_points.remove([x,y])
                break
        counts=counts+1
    
    for i in range(len(line)-1, -1, -1):
        if len(line[i]) <=10:
            line.pop(i)
            
    #Find curve at each line
    intensity_c = []
    curve_vessel = np.zeros((height, width))
    for i in range(len(line)):
        intensity_c.append([])
        j=0
        while j+6<len(line[i]):
            x1,y1 = line[i][j][:2]
            x,y = line[i][j+3][:2]
            x2,y2 = line[i][j+6][:2]
            theta1 = math.atan2(y-y1, x-x1)
            theta2 = math.atan2(y2-y, x2-x)
            theta = np.abs(theta1 - theta2)
            intensity = 255*(theta/np.pi)
            intensity_c[i].append(intensity)
    #        cv2.circle(curve_vessel, (x,y), 3, (intensity,intensity,intensity), -1)
            curve_vessel[y,x]= intensity
            j+=1
        intensity_c[i].sort()       
    cv2.imwrite('curve_vessel.jpg', curve_vessel)
    
    #######Radial Gradient
    radialGra_pos = []
    for i in range(height):
        for j in range(width):
            if heuristic[i,j]==255:
                radialGra_pos.append([i,j])
    
    for i in range(len(radialGra_pos)):
        heuristic[radialGra_pos[i][0],radialGra_pos[i][1]] = green[radialGra_pos[i][0],radialGra_pos[i][1]]
    
    ##substract gradient image by vessel
#    kernel = np.ones((5,5),np.uint8)
#    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i,j] == 1:
                heuristic[i,j] = 0
    cv2.imwrite('gradient.jpg',heuristic)
    
    ### calculate to find cup
    #from INTENSITY
    gradiant_pos = []
    for i in range(int(360/mum)):
        gradiant_pos.append([])
    center = [int(row/2),int(col/2)]
    i=0
    count = 0
    while(i<center[1]-50):
        sampling_posO = [center[0],center[1]+i+30]
        count = 0
        for j in range(0,360,mum):
            theta = np.radians(j)
            x,y = rotate(center, sampling_posO, theta)
            y = round(y)
            x = round(x)
            gradiant_pos[count].append([y,x])
            count+=1
        i+=5
    
    diffIntens_pos = []
    update = 0
    for i in range(len(gradiant_pos)):
        count = []
        for j in range(len(gradiant_pos[i])-3):
            if heuristic[gradiant_pos[i][j][0],gradiant_pos[i][j][1]]!=0 and heuristic[gradiant_pos[i][j+3][0],gradiant_pos[i][j+3][1]]!=0:
                diff = np.abs(heuristic[gradiant_pos[i][j][0],gradiant_pos[i][j][1]] - heuristic[gradiant_pos[i][j+3][0],gradiant_pos[i][j+3][1]])
                count.append([diff, gradiant_pos[i][j][0], gradiant_pos[i][j][1]])
        count.sort()
        if len(count)>5:
            select = count[len(count)-1]
            updateROW = select[1]
            updateCOL = select[2]                
            diffIntens_pos.append([updateROW,updateCOL])
    
    distance = []
    center = [int(row/2),int(col/2)]
    for i in range(len(diffIntens_pos)):
        dist = np.sqrt((diffIntens_pos[i][0]- center[0])**2+(diffIntens_pos[i][1]- center[1])**2)
        distance.append(dist)
    
    distance.sort()
#    mean = statistics.mean(distance)
    
    #from Blood vessel
    Cves_intens = []
    sum_intens = 0
    for i in range(curve_vessel.shape[0]):
        for j in range(curve_vessel.shape[1]):
            if curve_vessel[i,j]>0:
                Cves_intens.append(curve_vessel[i,j])
                sum_intens+=curve_vessel[i,j] 
    Cves_intens.sort()
    _,curve_vessel = cv2.threshold(curve_vessel,Cves_intens[int(len(Cves_intens)*0.7)],255,cv2.THRESH_BINARY)
    
    ##---------------calculate and adjust cup boundary
    cupB_pos = diffIntens_pos.copy()
    #for i in range(curve_vessel.shape[0]):
    #    for j in range(curve_vessel.shape[1]):
    #        if curve_vessel[i,j]==255:
    #            dist = np.sqrt((i- center[0])**2+(j- center[1])**2)
    #            if np.abs(dist-mean)>20:
    #                curve_vessel[i,j]=0
    #            else: cupB_pos.append([i,j])
    
    #for i in range(len(cupB_pos)):
    #    dist = np.sqrt((cupB_pos[i][0]- center[0])**2+(cupB_pos[i][1]- center[1])**2)
    #    theta = math.atan2(cupB_pos[i][0]- center[0],cupB_pos[i][1]- center[1])
    #    if np.abs(dist-mean)>15:
    #        cupB_pos[i][1] + np.abs(mean-dist)*0.3*np.cos(theta)
    #        cupB_pos[i][0] + np.abs(mean-dist)*0.3*np.sin(theta)
    
    ##--------------- drawing cup
    canvas = np.zeros((row,col))
    for i in range(len(cupB_pos)):
        if len(cupB_pos[i])>0:
    #        cv2.circle(img, (cupB_pos[i][1], cupB_pos[i][0]), 1, (255, 255, 255), 1)
            cv2.circle(heuristic, (cupB_pos[i][1], cupB_pos[i][0]), 1, (255, 255, 255), 1)        
            cv2.circle(canvas, (cupB_pos[i][1], cupB_pos[i][0]), 3, (255, 255, 255), -1)
    
    kernel = np.ones((5,5),np.uint8)
    curve_vessel = cv2.dilate(curve_vessel,kernel,iterations = 1)
    canvas = cv2.bitwise_or(canvas, curve_vessel)
    canvas = canvas.astype('uint8')
    circles = cv2.HoughCircles(canvas,cv2.HOUGH_GRADIENT,1,250,
                                param1=1,param2=2,minRadius=30,maxRadius=150)
    error = 'error_none'
    
    if len(circles)==1:
        cv2.circle(canvas ,(circles[0][0][0],circles[0][0][1]),circles[0][0][2],(255,255,255),1)
        cv2.circle(image_result ,(circles[0][0][0],circles[0][0][1]),circles[0][0][2],(255,0,0),1)
    else:
        print("circles not found or more than 2")
        error = 'circles not found or more than 2'
    ## add all together = heuristic image
    #heuristic = cv2.add(curve_vessel, heuristic)
    #cv2.imwrite('heuristic.jpg',heuristic)
    thresh = thresh*255
    
    center = (circles[0][0][0],circles[0][0][1])
    radious = circles[0][0][2]
    area = np.pi*radious**2
    
    return center, radious, area, error, image_result