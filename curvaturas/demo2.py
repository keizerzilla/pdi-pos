import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	
	imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	imgray = cv2.GaussianBlur(imgray, (3, 3), 0)
	ret, tresh = cv2.threshold(imgray, 127, 255, 0)
	contours, _ = cv2.findContours(tresh,
		                           cv2.RETR_TREE,
		                           cv2.CHAIN_APPROX_SIMPLE)
	im2 = frame.copy()
	cv2.drawContours(im2, contours, -1, (0, 255, 0), 4)
	
	cv2.imshow('frame', im2)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

