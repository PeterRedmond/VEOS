import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
cap = cv2.VideoCapture(1)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # Draw the face mesh annotations on the image.
        mp_drawing.draw_landmarks(
            image, 
            face_landmarks, 
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1))

        # Draw the left and right eye center points
        left_eye_center = face_landmarks.landmark[159]
        right_eye_center = face_landmarks.landmark[145]

        cv2.circle(image, 
                   (int(left_eye_center.x * image.shape[1]), 
                    int(left_eye_center.y * image.shape[0])), 
                   radius=4, 
                   color=(0, 0, 255), 
                   thickness=-1)

        cv2.circle(image, 
                   (int(right_eye_center.x * image.shape[1]), 
                    int(right_eye_center.y * image.shape[0])), 
                   radius=4, 
                   color=(0, 0, 255), 
                   thickness=-1)
        
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
