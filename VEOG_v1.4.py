import cv2
import mediapipe as mp
import multiprocessing
import matplotlib.pyplot as plt
from collections import deque
import time
import math
import numpy as np

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2):
        self.staticMode = staticMode
        self.maxFaces = maxFaces

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, landmark_drawing_spec=self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x,y])
                faces.append(face)
        return img, faces
    
 #   def right_eye_openness(self, face):
 #       left_point = (face[33][0], face[33][1])
 #       right_point = (face[133][0], face[133][1])
 #       center_top = self.midpoint(face[159][0], face[159][1], face[145][0], face[145][1])
 #       center_bottom = self.midpoint(face[157][0], face[157][1], face[153][0], face[153][1])

#        eye_width = math.hypot(left_point[0]-right_point[0], left_point[1]-right_point[1])
#        eye_height = math.hypot(center_top[0]-center_bottom[0], center_top[1]-center_bottom[1])

#        ratio = eye_height / eye_width

#        return min(1, ratio * 100) # normalize it

    def right_eye_openness(self, face):
        A = math.hypot(face[145][0] - face[159][0], face[145][1] - face[159][1])  # distance between vertical eye landmarks
        B = math.hypot(face[33][0] - face[157][0], face[33][1] - face[157][1])  # distance between other set of vertical eye landmarks
        C = math.hypot(face[133][0] - face[161][0], face[133][1] - face[161][1])  # distance between horizontal eye landmarks

        # calculate eye aspect ratio
        ear = (A + B) / (2.0 * C) 

        return ear

    def midpoint(self, p1_x, p1_y, p2_x, p2_y):
        return int((p1_x+p2_x)/2), int((p1_y+p2_y)/2)

# This function will run in a separate process
def plotter(plt_queue):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()  # Create a figure and an axes.

    # Create deques for storing data
    timestamps = deque(maxlen=10*60)
    eye_openness = deque(maxlen=10*60)

    while True:
        if not plt_queue.empty():
            right_eye_ratio = plt_queue.get()
            timestamps.append(time.time())  # Use the current time as our x value
            eye_openness.append(right_eye_ratio * 100)  # Multiply by 100 to get a percentage

            ax.clear()
            ax.plot(timestamps, eye_openness)
            ax.set_ylim([0, 100])  # Keep the y-axis scale constant
            plt.pause(0.001)  # Redraw the plot

def main(plt_queue):
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
    cap.set(cv2.CAP_PROP_FPS, 60)
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)

        if len(faces) > 0:
            right_eye_ratio = detector.right_eye_openness(faces[0]) # let's just look at the first face
            print(f'Right Eye Openness: {right_eye_ratio}')
            plt_queue.put(right_eye_ratio)  # Send the value to the plotter process

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    # Create a queue for communication
    plt_queue = multiprocessing.Queue()

    # Start the plotter process
    plot_process = multiprocessing.Process(target=plotter, args=(plt_queue,))
    plot_process.start()

    main(plt_queue)  # Pass the queue to the main function
