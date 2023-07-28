import cv2 #for image and video processing
import dlib  #library for ml and cv tasks,used for object detection,landmark detection. here used for detecting face and face landmarks particularly mouth and eyes
import numpy as np
from scipy.spatial import distance as dist #for calculating euclidean distance between certain points between eyes n mouth for ear n mar
from pygame import mixer #for alarming
import time
mixer.init()
sound = mixer.Sound('alarm.wav')



# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")#loads a pre-trained
#shape predictor
#model from a file called "shape_predictor_68_face_landmarks.dat".The shape predictor model is trained to predict the locations of 68 specific 
# facial landmarks or points on a face, such as the corners of the eyes, nose, 
# mouth, and other facial features. 

    

# Calculate eye aspect ratio (EAR) to detect drowsiness
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A=dist.euclidean(mouth[0],mouth[4])
    B=dist.euclidean(mouth[2],mouth[6])
    return (B/A)


# Load video stream
cap = cv2.VideoCapture(0)



# Initialize variables
frame_counter = 0
ALARM_ON = False



while True:
    ret, frame = cap.read() #ret is a boolean flag that will say if the frame was read correctly or not. 
    #frame is the actual frame that is read from the video.

    # Convert image to grayscale because It helps in simplifying algorithms and as well 
    # eliminates the complexities related to computational requirements. It makes room for
    # easier learning for those who are new to image processing. This is because grayscale 
    # compressors an image to its barest minimum pixel.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray, 0)

    for face in faces:
        # Detect landmarks on the face
        landmarks = predictor(gray, face)
        # print(f"{landmarks.part(36).x,landmarks.part(36).y}")
        

        # Extract left and right eye coordinates
        left_eye = []
        mouth=[]
        for i in range(36, 42):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        right_eye = []
        for i in range(42, 48):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(60,68):
            mouth.append((landmarks.part(i).x,landmarks.part(i).y))
            
        
        
        # Calculate EAR for left and right eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        mar=mouth_aspect_ratio(mouth);
       
        # Calculate average EAR
        ear = (left_ear + right_ear) / 2.0


        # Detect drowsiness if EAR is below threshold for a certain number of frames
        if ear < 0.25 or mar>0.3:
            frame_counter += 1
            if frame_counter >= 30:
                if not ALARM_ON:
                    ALARM_ON = True
                    
                    try:
                     sound.play()
                     
                         # time.sleep(0.05)
                    except: # isplaying = False
                     pass
                    
                    # Start alarm or any other alert mechanism
        else:
            frame_counter = 0
            ALARM_ON = False
            sound.stop()

        # # Draw eyes and EAR on the frame
        # cv2.drawContours(frame, [np.array(left_eye)], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [np.array(right_eye)], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [np.array(mouth)], -1, (0, 255, 0), 1)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (20, 20), cv2.CALIB_CB_NORMALIZE_IMAGE, 0.75, (149, 49, 255), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (20, 100), cv2.CALIB_CB_NORMALIZE_IMAGE, 0.75, (149, 49, 255), 1)


    # Show the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
