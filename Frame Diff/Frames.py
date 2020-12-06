"""
Abedin Sherifi
RBE595
Project
"""

"""
This program reads a video and outputs frames
"""

"""
Imports
"""
import cv2 
 
def frames_from_video(video_loc): 
    
    video = cv2.VideoCapture(video_loc)
    count = 0
    success = 1
  
    while success:
        success, image = video.read()
        cv2.imwrite("frame%d.jpg" % count, image)
        count += 1

if __name__ == '__main__':
    frames_from_video("Video_16.mp4") 

