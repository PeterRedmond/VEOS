import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import collections
import threading
import time
import winsound


class VideoCapture:
    def __init__(self, src=2):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.fps = 0
        self.last_update = time.time()

        def update_frame(cap):
            while self.running:
                self.ret, self.frame = cap.read()
                now = time.time()
                # Update FPS approximately every second
                if now - self.last_update > 1.0:
                    self.fps = cap.get(cv2.CAP_PROP_FPS)
                    self.last_update = now

        self.thread = threading.Thread(target=update_frame, args=(self.cap,))
        self.thread.start()

    def read(self):
        return self.ret, self.frame.copy(), self.fps

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize matplotlib figure for live graph
fig, ax = plt.subplots()

# Deque data structure to store the last 10 seconds of data, assuming 60 frames per second
openness_percentages = collections.deque(maxlen=600)


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
cap = VideoCapture()

# Calibration for open and closed eyes
print("Keep your eyes fully open for a few seconds for calibration.")
winsound.Beep(440, 500)
time.sleep(3)  # Sleep for 3 seconds
open_ears = []
for _ in range(60):
    ret, frame, fps = cap.read()
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
        break  # Process only the first detected face for calibration
    time.sleep(1/60)  # Ensure we're running at approximately 60fps
max_ear = np.mean(open_ears)

print("Now, close your eyes for a few seconds for calibration.")
winsound.Beep(440, 500)
time.sleep(3)  # Sleep for 3 seconds
closed_ears = []
for _ in range(60):
    ret, frame, fps = cap.read()
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
        break  # Process only the first detected face for calibration
    time.sleep(1/60)  # Ensure we're running at approximately 60fps
min_ear = np.mean(closed_ears)
print("Calibration complete")
winsound.Beep(440, 500)

# Disable matplotlib interactive mode for faster plot updates
plt.ioff()

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# Normal processing
while True:
    ret, frame, fps = cap.read()
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

   

    # Update FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Draw FPS on frame
    cv2.putText(frame, "FPS: {:.2f}".format(
        fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 # Display frame with FPS
    cv2.imshow("Frame", frame)
    # Display live graph
    plt.cla()
    plt.plot(list(openness_percentages))
    plt.title("Eye Openness Over Time (Last 10 seconds)")
    plt.ylim([0, 100])
    plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
