import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from keras.models import load_model
import keras.utils as ku
import matplotlib.pyplot as plt
from flask import Flask,render_template,Response,jsonify
# render_template-html intergration
# Flask-server
# Resposne-to the server
aux=None

value=""

app = Flask(__name__,template_folder='templates',static_folder="Static")

# '/' - Home/main page
@app.route('/')
def index():
    return render_template('index_main.html')

@app.route('/playground')
def playground():
    return render_template('PlayGround.html')

def gen():
    global value
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
    
    aux=cap # global varible later utilised in main to close cam when tab is closed

    det = HandDetector(maxHands=1)

    offset = 20 # to capture whole hand

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


            cv2.imwrite("demo.jpg",iw)
            imge = ku.load_img("demo.jpg", target_size=(300,300))
            ans = Recog(imge)
            textt=chr(65+ans.index(max(ans)))
            cv2.putText(img,textt,(25,50),cv2.FONT_HERSHEY_SIMPLEX,1,(170,255,0),2,cv2.LINE_AA)
            value=textt
        else:
            value=""
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpg\r\n\r\n'+frame+b'\r\n\r\n')
        time.sleep(0.04)

@app.route("/result")
def result():
    return jsonify(value)

@app.route("/video_feed")
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000',debug=True)
    # debug==true cause changes made can be reversible and site can be refreshed immediately (if mistakes r often )
    aux.release()
    #to close cam after tab close