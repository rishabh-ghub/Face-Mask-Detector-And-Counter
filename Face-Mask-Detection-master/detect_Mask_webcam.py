from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import pandas as pd
import datetime
import time
import os.path
from os import path
#   from extract_face import e
####

xtract_face
# load model
model = load_model('mask_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)
#webcam = r"test.jpg"
    
classes = ['without_mask','with_mask']
#i = 0
#j = 0
# loop through frames
while webcam.isOpened():
    i = 0
    j = 0

    # read frame from webcam
    status, frame1 = webcam.read()
    status, frame = webcam.read()
    

    # apply face detectionpp
    face, confidence = cv.detect_face(frame)

    cv2.rectangle(frame,(15,15),(623,463),(255,255,255), 1)


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        faceTrc = cv2.rectangle(frame, (startX,startY), (endX,endY), (255,0,0), 2)
        

        # crop the detected face region
        face_crop1 = np.copy(frame[startY:endY,startX:endX])
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label0 = classes[idx]

        label = "{}: {:.2f}%".format(label0, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10


        
        # write label and confidence above face rectangle
        if label0 == "with_mask" :
            i = i + 1 
            cv2.putText(frame, label + str(i), (startX-5, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 255, 0), 2)
            #cv2.putText(frame, "With Mask:" + str(i) , (10,300),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241,241,249), 2)#BGR
        else:
            j = j + 1
            cv2.putText(frame, label + str(j), (startX-5, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 0, 255), 2)
            cv2.imwrite(filename='img_.jpg', img = face_crop1)
            #cv2.putText(frame, "Without Mask:" + str(j) , (10,350),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241,241,249), 2)
            #extract_face(image_path = image_path)
            
        
        cv2.putText(frame, "With Mask:" + str(i) , (17,300),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, "Without Mask:" + str(j) , (17,350),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        #cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        ct = datetime.datetime.now()
        name1 = ct.strftime("%d_%m_%Y")
        if key == ord("p"):
            #print("With Mask:",i,"Student")
            #print("Without Mask:",j,"Student")
            dict = {'With Mask': i, 'Without Mask': j, 'Date': ct.strftime("%d-%m-%Y"), 'Time': ct.strftime("%H:%M:%S")}
            df = pd.DataFrame(dict,index=[0])
            #df.to_csv('Attendence.csv',mode = 'a',header = False)
            name = ct.strftime("%d_%m_%Y_%H_%M_%S")
            #cv2.imwrite(filename='Capture/Cam/img_'+ name +'.jpg', img = frame)
            cv2.imwrite(filename='Images/img_'+ name +'.jpg', img = frame1)
            
            #if label0 == "without_mask":
            
            if path.exists("Image_input/" + name1) == False:
                os.mkdir("Image_input/" + name1)
            cv2.imwrite(filename='Image_input/' + name1 + '/Img_'+ name +'.jpg', img = frame1)
            print("Image Saved @",ct,"\n")      

        if key == ord('m'):
            # release resources
            if path.exists("Image_input/" + str(name1)) == False:
                print("No Captures")
                break
            webcam.release()
            time.sleep(2)
            cv2.destroyAllWindows()
            exec(open('Image_Mask_Detection.py').read())
            
    #exec(open('Face_Recognition.py').read())
    # display output
    cv2.imshow("mask detection", frame)
    key = cv2.waitKey(1) & 0xFF
    # press "Q" to stop
    if key == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break

