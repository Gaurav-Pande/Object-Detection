# Object-Detection
Deep learning project

YOLO object detection in pytorch.
Things need to be done:
* Make it minimalistic
* add real time object detection 
* add mobile support

References: Pytorch-Yolov3

Object detection is a subset in computer vision which deals with automatic methods for identifying objects of interests in an image with respect to the background. In this repo we will implement and understand how YOLOv3 works.

# YOLOv3

1. Divide the image into multiple grids.
2. Each cell gives 3 boxes in feature maps like shown in the figure: b1,b2,b3. These boxes are chosen using k means algorithm   considering the scale, size, aspect ratio into consideration.
3. Each box is responsible for spitting out 3 things: 
  1. Box cordinates: tx,ty,tw,th (tx,tx = x,y cordinate of the top leftmost cell, tw,th = width and height of the box) which
     is the probable box containing the object.
  2. Objectness score represents the probability that an object is contained inside a bounding box. It should be nearly 1 for      the red and the neighboring grids, whereas almost 0 for, say, the grid at the corners.
  3. Class confidences represent the probabilities of the detected object belonging to a particular class (Dog, cat, banana,        car etc). Before v3, YOLO used to softmax the class scores.
