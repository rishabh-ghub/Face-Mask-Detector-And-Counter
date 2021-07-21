# FaceMaskDetector 
Since we know, that already a lot of work has been done in the field of face recognition, we wanted to do something new, therefore we integrated real time mask detection with our project.
In this, we capture the images in real time and detect whether they are with or without mask.

## Handle Addresses
Image_Mask_Detection.py (L-22) : 
>>[f for f in glob.glob(r'Address Of Image_input/' + name1 + "/**/*", recursive = True) if not os.path.isdir(f)]

Face_Recognition.py (L-37) :
>>[f for f in glob.glob(r'Address Of NoMaskStd/'+ str(name2)

## How to Run
- Step 0) Download project on your system.
- Step 1) Install all the dependencies from requirements.txt, using command prompt ' pip install -r requirements.txt '.
- Step 2) Run detect_Mask_webcam.py file, now you'll see realtime mask recognition with counter.
- Step 3) Press "p" to capture image, press "m" to trigger the whole process, press "q" to quit.

Note: After pressing "m" whole process is triggered automatically, you don't need to run Image_Mask_Detection.py, and Face_Recognition.py separately.

## Output stored in directories:
- Capture : Mask Detection with Counter.
- NoMaskStd : Non-masked faces.
- Known : Non-masked faces recognized through dataset.
- Unknown : Non-masked faces not present in dataset.


Note : - All the captures in each directories are saved in a different folder everyday.
       - All the captures of the day are processed in eack run.

## DataBase 
Attendence.csv : Stores the Counter value of each image, with Date & time.

No_Mask_Std.csv : Stores names of non-masked faces recognized by system, with Date & time.

## Loss & Accuracy after training:
- Loss: 0.0233 
- Accuracy: 0.9928 
- Validation loss: 0.0505
- Validation accuracy: 0.9868

### PreTrained Model : 
- Link for downloading the Model : https://drive.google.com/drive/folders/1bh-MMlvRwwhIl78Mqq1KUBzCdWYw_hbC?usp=sharing
- 'mask_detection.model' trained on the Sequential CNN model (trainin file not uploaded due to copyright issues).
- Paste this file in the folder.

Note : To run on your self trained model make sure to Edit the code in to read the model in  detect_Mask_webcam.py, Image_Mask_Detection.py files.

### Dataset :
- For mask detection : https://www.kaggle.com/dhruvmak/face-mask-detection
- For face recognition : self made, feel free to add more faces. Just make sure to name each image ;)  


# Feel free to play with the project make your own versions, run on your trained models mix and match !! Have fun and Learn :)




