"""
Pose detection
1. Keypoint detection  
2. Image Detection 
3. Image Classification 
4. Object Detection  
"""

import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

"""
Initialize the Pose Detection Model
The first thing that we need to do is initialize the pose class using the mp.solutions.pose syntax and then we will call the setup function mp.solutions.pose.Pose() with the arguments:

static_image_mode – It is a boolean value that is if set to False, the detector is only invoked as needed, that is in the very first frame or when the tracker loses track. If set to True, the person detector is invoked on every input image. So you should probably set this value to True when working with a bunch of unrelated images not videos. Its default value is False.
min_detection_confidence – It is the minimum detection confidence with range (0.0 , 1.0) required to consider the person-detection model’s prediction correct. Its default value is 0.5. This means if the detector has a prediction confidence of greater or equal to 50% then it will be considered as a positive detection.
min_tracking_confidence – It is the minimum tracking confidence ([0.0, 1.0]) required to consider the landmark-tracking model’s tracked pose landmarks valid. If the confidence is less than the set value then the detector is invoked again in the next frame/image, so increasing its value increases the robustness, but also increases the latency. Its default value is 0.5.
model_complexity – It is the complexity of the pose landmark model. As there are three different models to choose from so the possible values are 0, 1, or 2. The higher the value, the more accurate the results are, but at the expense of higher latency. Its default value is 1.
smooth_landmarks – It is a boolean value that is if set to True, pose landmarks across different frames are filtered to reduce noise. But only works when static_image_mode is also set to False. Its default value is True.
Then we will also initialize mp.solutions.drawing_utils class that will allow us to visualize the landmarks after detection, instead of using this, you can also use OpenCV to visualize the landmarks.
"""

#initialize mediapipe pose class
mp_pose = mp.solutions.pose

#setting up the pose function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

#Initializing mediapipe drawing class, useful for annotation
mp_drawing = mp.solutions.drawing_utils
connection_color = (0, 255, 0)  # Set to green (R=0, G=255, B=0)

#read an image from the specified path
sample_img = cv2.imread('sample.jpg')

#specific a size of the figure
plt.figure(figsize = [10, 10])

#display the sample image, also convert BGR to RGB for display
plt.title("sample Image"); plt.axis('off'); plt.imshow(sample_img[:,:,::-1]); plt.show()

"""
Perform Pose Detection
Now we will pass the image to the pose detection machine learning pipeline by using the function mp.solutions.pose.Pose().process(). But the pipeline expects the input images in RGB color format so first we will have to convert the sample image from BGR to RGB format using the function cv2.cvtColor() as OpenCV reads images in BGR format (instead of RGB).

After performing the pose detection, we will get a list of thirty-three landmarks representing the body joint locations of the prominent person in the image. Each landmark has:

x – It is the landmark x-coordinate normalized to [0.0, 1.0] by the image width.
y: It is the landmark y-coordinate normalized to [0.0, 1.0] by the image height.
z: It is the landmark z-coordinate normalized to roughly the same scale as x. It represents the landmark depth with midpoint of hips being the origin, so the smaller the value of z, the closer the landmark is to the camera.
visibility: It is a value with range [0.0, 1.0] representing the possibility of the landmark being visible (not occluded) in the image. This is a useful variable when deciding if you want to show a particular joint because it might be occluded or partially visible in the image.
"""

#perform pose detection after converting the image into RGB format
results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

#check if any landmarks are found
if results.pose_landmarks:
    #iterate two times as we only to display first two landmarks
    for i in range(2):
        #display the found normalized landmarks
        print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}') 

#now we will convert the two normalized landmarks displayed above into their original scale by using the width and height of the image
#retrieve the height and width of the sample image
image_height, image_width, _ = sample_img.shape

#check if any landmarks are found
if results.pose_landmarks:
    #iterate two times as we only want to display first two landmark
    for i in range(2):
        #display the found landmarks after converting them into their original scale
        print(f'{mp_pose.PoseLandmark(i).name}:') 
        print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
        print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')
        print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')
        print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')

#Now we will draw the detected landmarks on the sample image using the function mp.solutions.drawing_utils.draw_landmarks() and display the resultant image using the matplotlib library.
img_copy = sample_img.copy()

#check if any landmarks are found:
if results.pose_landmarks:
    #draw Pose landmarks on the sample image
    mp_drawing.draw_landmarks(image=img_copy, 
                            landmark_list=results.pose_landmarks, 
                            connections=mp_pose.POSE_CONNECTIONS, 
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=connection_color, thickness=2))

    #specify a size of the figure 
    fig = plt.figure(figsize = [10,10])
    #Display the output image with the landmarks drawn, also convert BGR to RGB for display
    plt.title("Output"); plt.axis('off'); plt.imshow(img_copy[:,:,::-1]); plt.show()

"""
Now we will go a step further and visualize the landmarks in three-dimensions (3D) using the function mp.solutions.drawing_utils.plot_landmarks(). We will need the POSE_WORLD_LANDMARKS that is another list of pose landmarks in world coordinates that has the 3D coordinates in meters with the origin at the center between the hips of the person.
"""
# Plot Pose landmarks in 3D.
mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)