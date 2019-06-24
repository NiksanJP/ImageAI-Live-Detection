# ImageAI-Live-Detection
Simple detection technique using ImageAI. This uses darknet's YOLOv3 which is only compatible with Linux and MAC. I decided to use Opencv and open each frame and process it with Image AI image detection.

# Installation
  install Image AI
  download YOLOv3 <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5"> here </a>
  Then add it to the file where the ImageAI.py file
  
  Make sure to create a images file to store the pictures that will be displayed to opencv
  
  Your File should look like this
  ObjectDetectionFile (TOP)
      Images (sub)
          pic.png
      imageAI.py
      yolo.h5
  
# What happens in the Program
  Nothing just that we process every frame that the camera captures to the image detection API.
  
 
