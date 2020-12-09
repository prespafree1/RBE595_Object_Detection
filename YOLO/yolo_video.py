"""
RBE595, Final Project: YOLO Algorithm Video Processor

Run YOLO algorithm on provided video file
Source: https://pjreddie.com/darknet/yolo/

@author: Jesse Morzel
"""

"""
Define imports
"""
import subprocess
from subprocess import Popen, PIPE, STDOUT
import cv2
import os
import time


"""
Define global variables
"""
thresh = '0.25'       # Threshold used for YOLO algorithm


def main():

    # Start execution timer
    start_time = time.time()

    # Read input video
    vid = cv2.VideoCapture('plane_16.avi')
    frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Frame count = %s' % frame_count)

    # Init output file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_vid = cv2.VideoWriter('output_video.avi', fourcc, 25, (720, 540))

    for i in range(int(frame_count) - 1):
        print('Frame count = %s' % vid.get(cv2.CAP_PROP_POS_FRAMES))

        ret, frame = vid.read()
        fstring = "frame%d.jpg" % i

        cv2.imwrite(fstring, frame)

        subprocess.run(["./darknet detect cfg/yolov3.cfg yolov3.weights %s -thresh %s" % (fstring, thresh)], shell=True)

        # Rename processed image file and add to output video
        os.rename(r'predictions.jpg',r'out%d.jpg' % i)
        out_frame = cv2.imread('out%d.jpg' % i)

        # Add threshold to video
        cv2.putText(out_frame, 'Th ' + thresh, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

        output_vid.write(out_frame)

    vid.release()
    output_vid.release()

    # End timer
    end_time = time.time()
    exec_time = end_time - start_time
    print('Execution Time = %d' % exec_time)


if __name__ == "__main__":
    main()
