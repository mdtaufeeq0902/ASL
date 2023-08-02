import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
from keras.models import load_model

cap = cv2.VideoCapture(0) # '0' is id of web cam being used

det = HandDetector(maxHands=1)

offset = 20 # to capture whole hand

# name of starting folder pertaining to ASL sign
name=ord('100')

#counter 
n=0

# path to data directory

fold="C:/Users/User/OneDrive/Desktop/Miniprj/images"
imsize=300 # fixed sized image used here


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

        cv2.imshow("Cropped",imgcrp)
        cv2.imshow("OnWhite",iw)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
    # 1 ms delay and waits for key interupt
    key = cv2.waitKey(1)
    
    # press z to satify below condition
    if key==ord('z'):
        # counter 'n'
        print(n)
        
        # path to directory where data can be stored  
        path = f"{fold}/{chr(name)}"

        if not os.path.exists(path):
            os.mkdir(path)
        cv2.imwrite(f"{path}/image_{n}.jpg",iw)
        
        n+=1
        # to take 100 imgs of all signs
        if(n==199):
            print("done for",chr(name))
            n=0
            name+=1