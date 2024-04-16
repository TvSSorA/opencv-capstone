import cv2 as cv

cap = cv.VideoCapture(1)

while True:
    scuccess, img = cap.read()

    cv.imshow("Image", img)
    cv.waitKey(1)
