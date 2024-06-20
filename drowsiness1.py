import pygame
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
import numpy as np
import time

# Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('alert.mp3')  # Update 'alert.mp3' with the correct path

# Minimum threshold of eye aspect ratio below which alarm is triggered
EYE_ASPECT_RATIO_THRESHOLD = 0.25

# Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 15

# Minimum threshold of mouth opening ratio below which alarm is triggered
MOUTH_OPENING_RATIO_THRESHOLD = 0.6  # Adjust as needed

# Minimum consecutive frames for which mouth ratio is above threshold for yawn to be detected
MOUTH_OPENING_RATIO_CONSEC_FRAMES = 7  # Adjust as needed

# Count no. of consecutive frames below threshold value for eyes and mouth
EYE_COUNTER = 0
MOUTH_COUNTER = 0

# Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MOR Extraction: mouth opening ratio
def mouth_opening_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mor = (A + B) / (2.0 * C)
    return mor

# This function calculates and returns eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2 * C)
    return ear

# Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extract indexes of facial landmarks for the left and right eye, and for the mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

# Start webcam video capture
video_capture = cv2.VideoCapture(0)

# Give some time for the camera to initialize (not required)
time.sleep(2)

while True:
    # Read each frame, flip it horizontally, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Image preprocessing: equalize histogram, Gaussian blur
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect facial points through detector function
    faces = detector(gray, 0)

    # Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around each face detected
    for (x, y, w, h) in face_rectangle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect facial points
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Get array of coordinates of leftEye, rightEye, and mouth
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # Calculate aspect ratio of both eyes and mouth
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)
        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        mouthOpeningRatio = mouth_opening_ratio(mouth)

        # Use hull to remove convex contour discrepancies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Detect if eye aspect ratio is less than threshold
        if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
            EYE_COUNTER += 1
        else:
            EYE_COUNTER = 0

        # Detect if mouth opening ratio is greater than threshold
        if mouthOpeningRatio > MOUTH_OPENING_RATIO_THRESHOLD:
            MOUTH_COUNTER += 1
        else:
            MOUTH_COUNTER = 0

        # If no. of frames is greater than threshold frames for eyes or mouth,
        if EYE_COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
            pygame.mixer.music.play(-1)
            cv2.putText(frame, "Drowsiness Detected", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        elif MOUTH_COUNTER >= MOUTH_OPENING_RATIO_CONSEC_FRAMES:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
            cv2.putText(frame, "Yawning Detected", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        else:
            pygame.mixer.music.stop()

    # Show video feed
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()

