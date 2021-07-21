import face_recognition
import cv2
import numpy as np
import datetime
import glob
import os
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
import os.path
from os import path

print("\nRecognizing Defaulters...\n ")
#video_capture = cv2.VideoCapture(0)
image_files1 = [f for f in glob.glob(r'DataSet' + "/**/*", recursive = True) if not os.path.isdir(f)]
list1 = []
for img1 in image_files1:      
    frame = face_recognition.load_image_file(img1)
    frame_encoding = face_recognition.face_encodings(frame)[0]
    list1.append(frame_encoding) 

# Create arrays of known face encodings and their names
known_face_encodings = list1
known_face_names = [f for f in listdir('DataSet') if isfile(join('DataSet', f))]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

ct1 = datetime.datetime.now()
name2 = ct1.strftime("%d_%m_%Y")
name3 = ct1.strftime("%d_%m_%Y_%H_%M_%S")

image_files = [f for f in glob.glob(r'C:\Users\user\Desktop\Face-Mask-Detection-master\NoMaskStd/'+ str(name2) + "/**/*", recursive = True) if not os.path.isdir(f)]
#while True:
i = 1
j = 1
for img in image_files:
    # Grab a single frame of video
    #ret, frame = video_capture.read()
    frame = cv2.imread(img)

    # Resize frame of video to 1/4 size for faster face recognition processing
    #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = cv2.resize(frame, (0, 0), fx=1.00, fy=1.00)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    #if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            ct = datetime.datetime.now()
            dict = {'No:': str(j), 'Name': name[:-4], 'Date': ct.strftime("%d-%m-%Y"),'Time': ct.strftime("%H-%M-%S")}
            j = j + 1
            df = pd.DataFrame(dict,index = [0])
            df.to_csv('No_Mask_Std.csv',mode = 'a',header = False)
            print(str(j - 1) + ") " + name[:-4] + " " + str(i))
            #print(name + str(i))
            face_names.append(name)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                #Scale back up face locations since the frame we detected in was scaled to 1/4 size
                #top *= 1
                #right *= 1
                #bottom *= 1
                #left *= 1
                # Draw a box around the face
                #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name[:-4], (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)
                if path.exists("Known/" + name2) == False:
                    os.mkdir("Known/" + name2)
                cv2.imwrite(filename='Known/' + name2 + '/Img_knw_'+ str(i) + " " + name3 +'.jpg', img = frame)           
        else:
            if path.exists("Unknown/" + name2) == False:
                os.mkdir("Unknown/" + name2)
                time.sleep(1)
            cv2.imwrite(filename='Unknown/' + name2 + '/Img_Unk_'+ str(i) + " " + name3 +'.jpg', img = frame)
            cv2.imshow('Unknown-' + str(i) , frame)
            
            

    process_this_frame = not process_this_frame
    i = i + 1



    # Display the results
    """for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name[:-4], (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)"""
    
    

    # Hit 'q' on the keyboard to quit!
if cv2.waitKey(1) & 0xFF == ord('q'):
    # Release handle to the webcam
    frame.release()
    cv2.destroyAllWindows()
