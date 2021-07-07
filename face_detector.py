#Import the cv2 library
import cv2

#Import the randrange function for different colors of rectangles 
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect the faces in
# img = cv2.imread('/Users/kevinjshah2207/Desktop/Projects/Face_Detector/image1.jpg')
# img = cv2.imread('image3.png')

#To capture video from webcam
webcam = cv2.VideoCapture(0)

#Iterate forever over frames
while True:
    ##Read the current frame
    successful_frame, frame = webcam.read()



    #Code to check of image is read properly

    # if img is None:
    #     print("Image is empty")

    #Convert image to grayscale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(img_gray)

    # print(face_coordinates)
    #Draw rectangles around faces
    for i in range(len(face_coordinates)):
        (x,y,w,h) = face_coordinates[i]
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
    
    #Display the image
    cv2.imshow('Face Detector Example', frame)

    # Key to keep image displayed so that we can see it until we press any button
    key = cv2.waitKey(1)

    #Press q or Q to exit the webcam
    if key == 81 or key == 113:
        break

#Release webcam
webcam.release()

print("Code Completed")