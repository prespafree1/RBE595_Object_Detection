# -*- coding: utf-8 -*-
"""
Derek Larson
RBE595
Final Project

Reference: https://github.com/Deetanshu/PyImageSearch
"""

#imports

import numpy as np
import argparse
import imutils
import cv2
import time
 
#Argument Parser made anaconda-friendly:
prototxt="MobileNetSSD_deploy.prototxt.txt"
cmodel="MobileNetSSD_deploy.caffemodel"
ap=argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="Minimum probability")
args=vars(ap.parse_args())

#Create list of classes and color set:
classes=["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
colors=np.random.uniform(0,255, size=(len(classes), 3))

#Load pre-trained model:
print("Loading Model: ")
net = cv2.dnn.readNetFromCaffe(prototxt, cmodel)

print("Starting to read video:")
video_inputs = cv2.VideoCapture('Video_16.mp4')
success,frame = video_inputs.read()
final_vid = []
num_of_objects = 0
num_of_all_obj = 0

text_file = open("Output.txt", "w")

#Loop for frames:
start = time.time()
while success:
    frame=imutils.resize(frame, width=400)
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections=net.forward()
    for i in np.arange(0, detections.shape[2]):
        num_of_all_obj += 1
        confidence=detections[0,0,i,2]
        if confidence > args["confidence"]:
            idx = int(detections[0,0,i,1])
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY)=box.astype("int")
            
            label="{}: {:.2f}%".format(classes[idx], confidence*100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
            y=startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
            num_of_objects += 1
            count_text = "Objects Detected %s" % str(num_of_objects)
            cv2.putText(frame, count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            text_file.write("Object " + str(num_of_objects) + ": " + str(classes[idx]) + " Confidence: " + str(round(confidence*100, 2)) + "%\n")
    #show images, within the bigger loop:
    final_vid.append(frame)
    success,frame = video_inputs.read()

end = time.time()
execution_time = end - start
text_file.write("Number of Objects above Confidence detected: " + str(num_of_objects) + "\n")
text_file.write("Number of total objects detected: " + str(num_of_all_obj) + "\n")
text_file.write("Execution time: " + str(execution_time) + "\n")
# set the video type and create .mp4 of the array at 10fps
text_file.close()
print("Compiling video:")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_vid = cv2.VideoWriter('final_video.mp4', fourcc, 10, (w, h))
for i in range(len(final_vid)):
    output_vid.write(final_vid[i])
output_vid.release()
print ("DONE")