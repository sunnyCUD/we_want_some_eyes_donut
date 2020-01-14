import numpy as np
import cv2
import math
from skimage.morphology import skeletonize

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
    validIntersection = [[0,1,0,1,0,0,1,0],
                         [0,0,1,0,1,0,0,1],
                         [1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],
                         [0,0,1,0,0,1,0,1],
                         [1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],
                         [1,0,1,0,0,1,0,0],
                         [0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],
                         [0,1,0,1,0,1,0,0],
                         [0,0,0,1,0,1,0,1],
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
    __,contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
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
    __,xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
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

def find_cup(imgROI,imgROI_disc):
    ########## INITIALIZING ##########
    blue,green,red = cv2.split(imgROI)
    row = height = imgROI_disc.shape[0]
    col = width = imgROI_disc.shape[1]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    green = clahe.apply(green)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51)), iterations = 1)
    
    #-------------------------------detect disc------------------------------------
    hsv = cv2.cvtColor(imgROI_disc, cv2.COLOR_BGR2HSV)
    
    ##### Blue
    lower_range = np.array([110,50,50])
    upper_range = np.array([130, 255,255])
    '''
    ##### green
    lower_range = np.array([40, 40,40])
    upper_range = np.array([70, 255,255])
    '''
    disc = cv2.inRange(hsv, lower_range, upper_range)
    
    image_result = imgROI.copy()
    center = [int(row/2), int(col/2)]
    __,contours, _ = cv2.findContours(disc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours is not None:
        hull= cv2.convexHull(contours[0])
        (center[0], center[1]), (MA, ma), angle = cv2.fitEllipse(hull)
        cv2.ellipse(disc, cv2.fitEllipse(hull), (255,255,255), thickness=cv2.FILLED)
        cv2.ellipse(image_result, cv2.fitEllipse(hull), (0,255,0), thickness=2)
    
    #------------------------------- SAMPLING IMAGE -------------------------------
    sampling = np.zeros((row, col))
    i=0
    mum = 4
    while(i<center[1]-80):
        
        sampling_posO = [center[0],center[1]+i+40]
        
        for j in range(0,360,mum):
            
            theta = np.radians(j)
            x,y = rotate(center, sampling_posO, theta)
            sampling_pos = [round(y),round(x)]
            sampling[sampling_pos[0],sampling_pos[1]] = green[sampling_pos[0],sampling_pos[1]]
            
        i+=4
        
    ##substract sampling by disc
    kernel = np.ones((15,15),np.uint8)
    disc_small = cv2.erode(disc,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(43,43)),iterations = 1)
    
    disc_small= disc_small.astype('float64')
    heuristic = cv2.bitwise_and(sampling, disc_small)
    
    #-------------------------- vessel curvature detection ------------------------
    ##vessel detection
    blood_vessels = extract_bv(imgROI)
    BV_forskeleton = cv2.medianBlur(blood_vessels,11)
    blood_vessels = cv2.dilate(blood_vessels,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),iterations = 1)
    
    blood_vessels = (255-blood_vessels)/255
    BV_forskeleton = (255-BV_forskeleton)/255
    
    ##Curve vessel
    skeleton = skeletonize(BV_forskeleton)
    skeleton = skeleton*255
    skeleton = skeleton.astype(np.uint8)
    disc_small = disc_small.astype(np.uint8)
    skeleton = cv2.bitwise_and(skeleton, disc_small)
    skeleton_copy = skeleton.copy()
    
    #store skeelton's white pixels
    points = []
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i,j]==255:
                points.append([j,i])
    
    intersections = getSkeletonIntersection(skeleton)
    for i in range(len(intersections)):
        cv2.circle(skeleton_copy, intersections[i], 2, (255,255,255), thickness=cv2.FILLED) #skeleton_copy to see where intercet and end points
        
    end_pointTU = skeleton_endpoints(skeleton)
    end_points = []
    for i in range(len(end_pointTU[0])):
        cv2.circle(skeleton_copy, (end_pointTU[1][i],end_pointTU[0][i]), 2, (255,255,255), thickness=cv2.FILLED)
        x = end_pointTU[1][i]
        y = end_pointTU[0][i]
        end_points.append([x,y])
    
    ##store all intersect points to end_point in [x,y] form
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
        if len(line[i]) <=20:
            line.pop(i)
            
    #Find curve at each line
    intensity_c = []
    curve_vessel = np.zeros((height, width))
    for i in range(len(line)):
        intensity_c.append([])
        j=0
        while j+6<len(line[i]):
            x1,y1 = line[i][j][:2]
            x,y = line[i][j+2][:2]
            x2,y2 = line[i][j+4][:2]
            theta1 = math.atan2(y-y1, x-x1)
            theta2 = math.atan2(y2-y, x2-x)
            theta = np.abs(theta1 - theta2)
            intensity = 255*(theta/np.pi)
            intensity_c[i].append(intensity)
            curve_vessel[y,x]= intensity
            j+=1
        intensity_c[i].sort()       
    
    ##substract gradient image by vessel
    kernel = np.ones((5,5),np.uint8)
    blood_vessels = cv2.dilate(blood_vessels,kernel,iterations = 1)
    for i in range(blood_vessels.shape[0]):
        for j in range(blood_vessels.shape[1]):
            if blood_vessels[i,j] == 1:
                heuristic[i,j] = 0
                
    ###------------------------- calculate to find cup ----------------------------
    ##------------from INTENSITY
    #gradiant_pos = postion at each rotation, count = at theta count
    gradiant_pos = []
    
    for i in range(int(360/mum)):
        gradiant_pos.append([])
    i=0
    while(i<center[1]-80):
        sampling_posO = [center[0],center[1]+i+40]
        count = 0
        for j in range(0,360,mum):
            theta = np.radians(j)
            x,y = rotate(center, sampling_posO, theta)
            y = round(y)
            x = round(x)
            gradiant_pos[count].append([y,x])
            count+=1
        i+=4
    
    intens = []
    for i in range(len(gradiant_pos)):
        for j in range(len(gradiant_pos[i])):
            intens.append(heuristic[gradiant_pos[i][j][0],gradiant_pos[i][j][1]])
    max_intens = np.amax(intens)
    
    diffIntens_pos = []
    for i in range(len(gradiant_pos)):
        count = []
        for j in range(len(gradiant_pos[i])-3):
            intensity_seed = heuristic[gradiant_pos[i][j][0],gradiant_pos[i][j][1]]
            intensity_iterate = heuristic[gradiant_pos[i][j+3][0],gradiant_pos[i][j+3][1]]
            if intensity_seed!=0 and intensity_iterate!=0:
                diff = np.abs(heuristic[gradiant_pos[i][j][0],gradiant_pos[i][j][1]] - heuristic[gradiant_pos[i][j+3][0],gradiant_pos[i][j+3][1]])
                count.append([diff, gradiant_pos[i][j][0], gradiant_pos[i][j][1]])
        count.sort()        ####combine all difference intensity then sort
        if len(count)>5:
            for i in range(len(count)-1,0,-1):
                if max_intens-heuristic[count[i][1],count[i][2]]<=40:
                    select = count[i]
                    updateROW = select[1]
                    updateCOL = select[2]                
                    diffIntens_pos.append([updateROW,updateCOL])
                    break
    
    ##------------from Blood vessel
    Cves_intens = []
    sum_intens = 0
    for i in range(curve_vessel.shape[0]):
        for j in range(curve_vessel.shape[1]):
            if curve_vessel[i,j]>0:
                Cves_intens.append(curve_vessel[i,j])
                sum_intens+=curve_vessel[i,j] 
    Cves_intens.sort()
    if Cves_intens!=[]:
        _,curve_vessel = cv2.threshold(curve_vessel,Cves_intens[int(len(Cves_intens)*0.92)],255,cv2.THRESH_BINARY)
    
    ##--------------- plot points
    cupB_pos = diffIntens_pos.copy()
    canvas = np.zeros((row,col))
    for i in range(len(cupB_pos)):
        if len(cupB_pos[i])>0:
            cv2.circle(heuristic, (cupB_pos[i][1], cupB_pos[i][0]), 2, (255, 255, 255), -1)        
            cv2.circle(canvas, (cupB_pos[i][1], cupB_pos[i][0]), 3, (255, 255, 255), -1)
#            cv2.circle(image_result, (cupB_pos[i][1], cupB_pos[i][0]), 2, (125, 125, 125), -1)      #for testing, care to remove in the future
    
    kernel = np.ones((5,5),np.uint8)
    curve_vessel = cv2.dilate(curve_vessel,kernel,iterations = 1)
    canvas = cv2.bitwise_or(canvas, curve_vessel)
    
#    for i in range(500):                    #for testing, care to remove in the future
#        for j in range(500):
#            if curve_vessel[i,j] == 255:
#                image_result[i,j][0] = 0
#                image_result[i,j][1] = 0
#                image_result[i,j][2] = 0
    
    ##--------------- draw circle on cup
    canvas = canvas.astype('uint8')
    
    circles = cv2.HoughCircles(canvas,cv2.HOUGH_GRADIENT,1,250,
                                param1=1,param2=1,minRadius=10,maxRadius=100)
    error = 'error_none'
    if circles is None:
        error = 'circles not found'
        center = error
        radious = error
        area = error
    else:
        circles = np.uint16(np.around(circles))
        count_set = []
        for i in circles[0,:]:
            mask_test = np.zeros((canvas.shape[0],canvas.shape[1]))
            count=0
            cv2.circle(mask_test,(i[0],i[1]),i[2],(255,255,255),2)
            for j in range(canvas.shape[0]):
                for k in range(canvas.shape[1]):
                    if mask_test[j,k]!=0 and mask_test[j,k]==canvas[j,k]:
                        count+=1
            count_set.append(count)
        most_fitting = np.amax(count_set)
        most_fitting_index = count_set.index(most_fitting)
        
    cv2.circle(canvas ,(circles[0][most_fitting_index][0],circles[0][most_fitting_index][1]),circles[0][most_fitting_index][2],(255,255,255),1)
    cv2.circle(image_result ,(circles[0][most_fitting_index][0],circles[0][most_fitting_index][1]),circles[0][most_fitting_index][2],(255,0,0),2)
    center = (circles[0][most_fitting_index][0],circles[0][most_fitting_index][1])
    radious = circles[0][most_fitting_index][2]
    area = np.pi*radious**2
    
    # add all together = heuristic image
    heuristic = cv2.add(curve_vessel, heuristic)
    blood_vessels = blood_vessels*255
    
    #print(error)
    
    return center, radious, area, error, image_result