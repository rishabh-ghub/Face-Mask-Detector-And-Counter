from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import pandas as pd
import datetime
import time
import glob
import os.path
from os import path
#   from extract_face import extract_face

# load model
model = load_model('mask_detection.model')

ct = datetime.datetime.now()
name = ct.strftime("%d_%m_%Y_%H_%M_%S")
name1 = ct.strftime("%d_%m_%Y")

if path.exists("Image_input/" + str(name1)) == False:
    print("No Captures")
    quit()
    
image_files = [f for f in glob.glob(r'C:\Users\user\Desktop\Face-Mask-Detection-master\Image_input/' + name1 + "/**/*", recursive = True) if not os.path.isdir(f)]
print("Analyzing Input..\n")
z = 1
for img in image_files:
    frame = cv2.imread(img)
    classes = ['without_mask','with_mask']
    i = 0
    j = 0
    face, confidence = cv.detect_face(frame)


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
            cv2.putText(frame, label + str(i) , (startX-5, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 255, 0), 2)
            
            #cv2.putText(frame, "With Mask:" + str(i) , (10,300),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241,241,249), 2)#BGR
        else:
            cv2.imwrite(filename='img_.jpg', img = face_crop1)
            ct1 = datetime.datetime.now()
            name2 = ct1.strftime("%d_%m_%Y_%H_%M_%S")
            name3 = ct1.strftime("%d_%m_%Y")
            j = j + 1
            if path.exists("NoMaskStd/" + name1) == False:
                os.mkdir("NoMaskStd/" + name1)
            cv2.imwrite(filename='NoMaskStd/'+ str(name3) +'/Img_'+ str(j)+ '_' + name2 +'.jpg', img = face_crop1)
            cv2.putText(frame, label + str(j), (startX-5, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 0, 255), 2)
            time.sleep(1)
                    
    cv2.putText(frame, "With Mask:" + str(i) , (10,300),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, "Without Mask:" + str(j) , (10,350),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    print("With Mask:",i,"Student")
    print("Without Mask:",j,"Student")
    #ct = datetime.datetime.now()
    dict = {'With Mask': i, 'Without Mask': j, 'Date': ct.strftime("%d-%m-%Y"), 'Time': ct.strftime("%H:%M:%S")}
    df = pd.DataFrame(dict,index=[0])
    df.to_csv('Attendence.csv',mode = 'a',header = False)
    #name = ct.strftime("%d_%m_%Y_%H_%M_%S")
    #name3 = ct.strftime("%d_%m_%Y")
    if path.exists("Capture/" + name1) == False:
        os.mkdir("Capture/" + name1)
    cv2.imwrite(filename='Capture/' + str(name1) + '/img_' + str(z) + "-"+ name +'.jpg', img = frame)
    z = z + 1
    print("Attendence Saved @",ct)      

print("\nData Saved!!")
# display output
#cv2.imshow("mask detection(L)", frame)
time.sleep(3)
exec(open('Face_Recognition.py').read())

#time.sleep(2)
#exec(open('Face_Recognition.py').read())
# press "Q" to stop
if cv2.waitKey(1) & 0xFF == ord('q'):
    frame.release()
    cv2.destroyAllWindows()




