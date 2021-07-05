#Import the cv2 library
import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect the faces in
# img = cv2.imread('/Users/kevinjshah2207/Desktop/Projects/Face_Detector/image1.jpg')
img = cv2.imread('image1.jpg')

#Code to check of image is read properly

if img is None:
    print("Image is empty")

#Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(img_gray)
print(face_coordinates)
#Draw rectangles around faces
(x,y,w,h) = face_coordinates[0]
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#Display the image
cv2.imshow('Face Detector Example', img)

# Key to keep image displayed so that we can see it until we press any button
cv2.waitKey()

print("Code Completed")