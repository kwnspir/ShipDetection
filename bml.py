import numpy as np
import cv2
import time
import os
from datetime import datetime

'''
Screenshot_2020-08-13_18-06-15.png
Screenshot_2020-08-13_18-06-40.png
Screenshot_2020-08-13_18-03-07.png
Screenshot_2020-08-13_18-02-42.png
Screenshot_2020-08-13_17-59-47.png
Screenshot_2020-08-16_18-26-49.png
Screenshot_2020-08-16_18-33-15.png
2523523552-1.jpg
'''


photo="Screenshot_2020-08-13_18-02-42.png"
#cascadefile="/cascades/cascade_p85_n300_s11.xml"
cascadefile="/data/cascade.xml"
save=0
os.chdir("./Saves") 
face_cascade = cv2.CascadeClassifier("/home/kayn/Desktop/Ship_Identify"+str(cascadefile))
first_par=float(input("Firssst par : "))
sec_par=int(input("Second par : "))
mins=int(input("min par : "))
maxs=int(input("max par : "))

while 1:
	#img=cv2.imread("/home/keyn/Desktop/Screenshot_2020-08-13_18-06-40.png")
	img = cv2.imread("./"+str(photo), cv2.IMREAD_GRAYSCALE)
	print(str(img))
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	try:
		faces = face_cascade.detectMultiScale(img, scaleFactor=first_par, minNeighbors=sec_par,minSize=(mins, mins),maxSize=(maxs,maxs))
	except Exception as e:
		print(str(e))
	#cv2.rectangle(img,(0,0),(100,100),(255,0,0),2)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		#roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]


	cv2.imshow('img',img)
	time.sleep(1)
	if save == 0:
		cv2.imwrite(str(photo)+"_new_"+str(datetime.now())+"_f_"+str(first_par)+"_s_"+str(sec_par)+".png", img)
		save=1

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cv2.destroyAllWindows()
