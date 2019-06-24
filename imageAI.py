from imageai.Detection import VideoObjectDetection
from imageai.Detection import ObjectDetection
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg 
import tensorflow as tf 
import numpy as np 
import PIL 
import cv2
import os 

class Detection:
    def __init__(self):
        self.path = os.getcwd()
        
        self.detector = VideoObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath("ObjectDetectionTensorflow/yolo.h5")
        self.detector.loadModel(detection_speed="fast")
        
        self.imgDetector = ObjectDetection()
        self.imgDetector.setModelTypeAsYOLOv3()
        self.imgDetector.setModelPath("ObjectDetectionTensorflow/yolo.h5")
        self.imgDetector.loadModel()
                
        self.camera = cv2.VideoCapture(0)
        
    def liveVideo(self):
        #Live to Video
        videoPath = self.detector.detectObjectsFromVideo(
            camera_input=self.camera,
            output_file_path=os.path.join(self.path, "Loaded_Video"),
            frames_per_second=30, log_progress=True, 
            minimum_percentage_probability=80
        )
        print(videoPath)
        cv2.imshow('video', videoPath)
        
    def liveVideoShow(self):      
        #Live video detection 
        while True:
            ret, frame = self.camera.read()
            
            img = PIL.Image.fromarray(frame)
            img.save("ObjectDetectionTensorflow/images/pic.png")
            
            detected = self.imgDetector.detectCustomObjectsFromImage(
                input_image="ObjectDetectionTensorflow/images/pic.png",
                output_image_path="ObjectDetectionTensorflow/images/pic.png",
                minimum_percentage_probability=40
            )
            
            for eachObject in detected:
                print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
                print("--------------------------------")
            
            if cv2.waitKey(33) == ord('a'):
                break
            
            img = mpimg.imread("ObjectDetectionTensorflow/images/pic.png")
            
            cv2.imshow('video', img)

detect = Detection()
detect.liveVideoShow()