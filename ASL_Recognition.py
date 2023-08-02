import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
from keras.models import load_model
import keras.utils as ku
import matplotlib.pyplot as plt

# Recognition function
def Recog(img):
    # Normalizing Image
    norm_img = ku.img_to_array(img)/255
    # Converting Image to Numpy Array
    input_arr_img = np.array([norm_img])
    # Getting Predictions
    pred = model.predict(input_arr_img)
    # Printing Model Prediction
    l=list(pred[0])
    return l

cap = cv2.VideoCapture(0) # '0' is id of web cam being used

det = HandDetector(maxHands=1)

offset = 20 # to capture whole hand

# name of starting folder pertaining to ASL sign
name=ord('0')

#counter 
n=0

imsize=300 # fixed sized image used here

# model deployment
model = load_model("Final_Model/ASL_Recog_model.h5",compile=False)

while True: 
    # success reps status of capture(1,0)
    success,img = cap.read()

    hands,img = det.findHands(img)
    if hands:
        hand = hands[0] # 0 reps single right/left hand  

        # bbox: boundary box
        x,y,wid,height = hand['bbox']

        # create an image so that all img are of ame sizes while testing/training+white backgrd
        iw = np.ones((imsize,imsize,3),np.uint8)*255 # '3' rep to store rgb,unint8-number standard(0,255) 
        
        #offset to capture whole image
        imgcrp = img[y-offset:y+height+offset,x-offset:x+wid+offset]

        # aspect ratio
        ar = height/wid

        if ar>1:
            k = imsize/height
            calwid = math.ceil(k*wid)  
            imgres = cv2.resize(imgcrp,(calwid,imsize))
            # center it cause white space is redundant
            wGap = math.ceil((imsize-calwid)/2)
            # map imgres to imwhite
            iw[0:,wGap:calwid+wGap]=imgres
        else:
            k = imsize/wid
            calh = math.ceil(k*height)
            imgres = cv2.resize(imgcrp,(imsize,calh))
            # center it cause white space is redundant
            hGap = math.ceil((imsize-calh)/2)
            # map imgres to imwhite
            iw[hGap:hGap+calh,0:]=imgres
        # cv2.imshow("Cropped",imgcrp)
        # cv2.imshow("OnWhite",iw)
        cv2.imwrite("demo.jpg",iw)
        imge = ku.load_img("demo.jpg", target_size=(300,300))
        ans = Recog(imge)
        textt=chr(65+ans.index(max(ans)))
        cv2.putText(img,textt,(25,50),cv2.FONT_HERSHEY_SIMPLEX,1,(170,255,0),2,cv2.LINE_AA)


    cv2.imshow("Image",img)
    cv2.waitKey(1)