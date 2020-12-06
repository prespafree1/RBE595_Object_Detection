"""
Abedin Sherifi
RBE 595
Project
"""

"""
Reference from https://www.tutorialfor.com/blog-285124.htm
"""

"""
Imports
"""
import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
import sys
import time

#Receive file names of the frames and sort them
list_frames = os.listdir('Frames/')
list_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

#Empty list of images
list_images=[]

for i in list_frames:
    image = cv2.imread('Frames/'+i)
    list_images.append(image)

#Kernel for image dilation
kernel = np.ones((4,4),np.uint8)

#Font style
font = cv2.FONT_HERSHEY_SIMPLEX

#Directory to save the ouput contours
pathIn = "Contours/"
f = open('Objects_Detected.txt', 'w+')
start_time = time.time()
for i in range(len(list_images)-1):
    
    #Frame differencing
    current_gray = cv2.cvtColor(list_images[i], cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(list_images[i+1], cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(next_gray, current_gray)
    
    #Image thresholding
    ret, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    #Image dilation
    dilated = cv2.dilate(thresh,kernel,iterations = 1)
    
    #Finding contours
    contours, h = cv2.findContours(dilated.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    #Contours appearing in the object detection ROI. Number of detected objects will be displayed on each contour frame.
    good_contours = []
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        if (x <= 700) & (y >= 20) & (cv2.contourArea(cont) >= 200):
            good_contours.append(cont)

    cont_list = list_images[i].copy()
    cv2.drawContours(cont_list, good_contours, -1, (255,0,255), 2)
    
    cv2.putText(cont_list, "Detected Objects: " + str(len(good_contours)), (500, 15), font, 0.5, (255, 0, 255), 1)
    cv2.line(cont_list, (0, 20),(700,20),(255, 0, 255), 2)
    cv2.imwrite(pathIn+str(i)+'.png',cont_list)
    f.write(f'Frame: {i} Objects Detected: {str(len(good_contours))} \r\n')

end_time = time.time()
solve_time = end_time - start_time
print(f'Object detection performed on whole video for {solve_time} seconds')
f.close()
#Video output specifications
pathOut = 'Object_Detection.mp4'
fps = 14.0

total_frames_array = []
contour_frames = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

contour_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

for i in range(len(contour_frames)):
    filename=pathIn + contour_frames[i]
    image = cv2.imread(filename)
    height, width, l = image.shape
    size = (width,height)
    total_frames_array.append(image)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(total_frames_array)):
    out.write(total_frames_array[i])

out.release()

