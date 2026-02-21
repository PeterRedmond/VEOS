import cv2
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path1 = filedialog.askopenfilename()
# Create a video capture object, in this case we are reading the video from a file
vid_capture1 = cv2.VideoCapture(file_path1)
file_path2 = filedialog.askopenfilename()
# Create a video capture object, in this case we are reading the video from a file
vid_capture2 = cv2.VideoCapture(file_path2)

if (vid_capture1.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
    if (vid_capture2.isOpened() == False):
        print("Error opening the video file")
    # Read fps and frame count
    else:

        # Get frame rate information
        # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        fps1 = vid_capture1.get(5)
        print('Frames per second 1 : ', fps1, 'FPS')

        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frame_count1 = vid_capture1.get(7)
        print('Frame count 1 : ', frame_count1)
        # # Get frame rate information
        # # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        fps2 = vid_capture2.get(5)
        print('Frames per second 2 : ', fps2, 'FPS')

        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frame_count2 = vid_capture2.get(7)
        print('Frame count 2 : ', frame_count2)
currFrame = 1
startFrame = 1680
cnt = 0
font = cv2.FONT_HERSHEY_DUPLEX
color = (255, 0, 0)  # red
fontsize = 2
text = "test"
position = (20, 50)

while(vid_capture1.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is frames
    ret, frame1 = vid_capture1.read()
    if ret == True:
        if (currFrame >= startFrame):
            cv2.putText(frame1, str(currFrame), position,
                        font, fontsize, color=color)
            cv2.imshow('Frame1', frame1)
            cnt = cnt+1
            if (cnt <= 3):
                ret, frame2 = vid_capture2.read()
                #ret, frame2 = vid_capture2.read()
                if ret == True:

                    cv2.putText(frame2, str(int((currFrame-startFrame-1)/1)), position,
                                font, fontsize, color=color)
                    cv2.imshow('Frame2', frame2)
            if (cnt == 4):
                cnt = 0

            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(45)
            if key == 32:
                cv2.waitKey()

            if key == ord('q'):
                break
        currFrame = currFrame+1
    else:
        break

# Release the video capture object
vid_capture1.release()
vid_capture2.release()
cv2.destroyAllWindows()
