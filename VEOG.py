import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import time
import winsound
import collections

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize matplotlib figure for live graph
fig, ax = plt.subplots()

# Deque data structure to store the last 10 seconds of data, assuming 30 frames per second
openness_percentages = collections.deque(maxlen=300)


def eye_aspect_ratio(eye):
    # Vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Horizontal eye landmark
    C = np.linalg.norm(eye[0] - eye[3])

    # Eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear


# Open webcam
cap = cv2.VideoCapture(1)
for _ in range(100):
    ret, frame = cap.read()

# Calibration for open and closed eyes
print("Keep your eyes fully open for a few seconds for calibration.")
winsound.Beep(440, 500)
time.sleep(3)  # Sleep for 3 seconds
open_ears = []
for _ in range(30):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y)
                            for i in range(36, 42)], np.int32)
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y)
                             for i in range(42, 48)], np.int32)
        open_ears.append((eye_aspect_ratio(left_eye) +
                         eye_aspect_ratio(right_eye)) / 2.0)
max_ear = np.mean(open_ears)

print("Now, close your eyes for a few seconds for calibration.")
winsound.Beep(440, 500)
time.sleep(3)  # Sleep for 3 seconds
closed_ears = []
for _ in range(30):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y)
                            for i in range(36, 42)], np.int32)
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y)
                             for i in range(42, 48)], np.int32)
        closed_ears.append((eye_aspect_ratio(left_eye) +
                           eye_aspect_ratio(right_eye)) / 2.0)
min_ear = np.mean(closed_ears)
print("Calibration complete")
winsound.Beep(440, 500)
# Normal processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y)
                            for i in range(36, 42)], np.int32)
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y)
                             for i in range(42, 48)], np.int32)

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        # Normalization to get percentage
        openness = (ear - min_ear) / (max_ear - min_ear) * 100
        openness = max(0, min(openness, 100))  # Ensuring it's within 0-100
        openness_percentages.append(openness)

        # Draw on frame
        radius = int(openness / 2)
        cv2.circle(frame, (landmarks.part(36).x,
                   landmarks.part(36).y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (landmarks.part(42).x,
                   landmarks.part(42).y), radius, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    # Display live graph
    plt.cla()
    plt.plot(list(openness_percentages))
    plt.title("Eye Openness Over Time (Last 10 seconds)")
    plt.ylim([0, 100])
    plt.pause(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
