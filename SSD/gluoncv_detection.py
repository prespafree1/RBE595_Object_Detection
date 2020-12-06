"""
Abedin Sherifi
RBE 595
GluonCV and MXNet
"""

"""
Reference used from https://github.com/dmlc/gluon-cv
"""

"""
Imports
"""
import gluoncv as gcv
gcv.utils.check_version('0.4.0')
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
import mxnet as mx
import time

#Loading a GCV already trained model
model = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)

#Loading video
cap = cv2.VideoCapture('Video_16.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('Object_Detection_GluonCV.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
f = open('Object_Detection_Per_Frame.txt', 'w+')
#f = open('Objects_Detected.txt', 'w+')

while True:
    #Load camera frames
    ret, frame = cap.read()
    
    #Image pre-processing
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    color_nd, out_frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=540, max_size=720)

    #Run frames through CNN
    class_ident, confidence, boxes = model(color_nd)
    
    #Display detected objects with confidence number, bounding box
    scale = 1.0 * frame.shape[0] / out_frame.shape[0]
    start_time = time.time()
    image = gcv.utils.viz.cv_plot_bbox(frame.asnumpy(), boxes[0], confidence[0], class_ident[0], class_names=model.classes, scale=scale)
    gcv.utils.viz.cv_plot_image(image)
    cl = class_ident.asnumpy()
    conf = confidence.asnumpy()
    #f.write(f'Objects detected per frame {cl[0][0][0]} with confidence {conf[0][0][0]}\r\n')
    end_time = time.time()
    solve_time = end_time - start_time
    f.write(f'Object detection performed on single frame for {solve_time}\r\n')
    out.write(image)
    if cv2.waitKey(1) & 0xff == ord('q'):
    	cv2.destroyAllWindows()
    	cap.release()


