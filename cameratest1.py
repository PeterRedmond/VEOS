import subprocess as sp
import numpy as np
import cv2
import time

# Camera settings
device_name = "Global Shutter Camera"  # Replace with your camera's exact name
width, height = 640, 480  # Lower resolution for testing
fps_target = 100  # Target high frame rate for testing

# FFmpeg command
ffmpeg_cmd = [
    "ffmpeg",
    "-f", "dshow",
    "-video_size", f"{width}x{height}",
    "-framerate", str(fps_target),  # Try to achieve this frame rate
    "-pixel_format", "yuyv422",  # Input pixel format
    "-i", f"video={device_name}",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",  # Output pixel format
    "-"
]

# Start FFmpeg process
process = sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=10**8)

# Calculate frame size in bytes (for YUYV422 -> BGR24)
frame_size = width * height * 3  # 3 bytes per pixel for bgr24

# Frame count and time tracking
frame_count = 0
start_time = time.time()

try:
    while True:
        # Read a single frame from FFmpeg's stdout
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            print(f"❌ Incomplete frame received: expected {frame_size} bytes, got {len(raw_frame)}.")
            continue  # Skip this incomplete frame and read again

        # Convert raw bytes to NumPy array (BGR24 format)
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

        # Display the frame (optional)
        cv2.imshow("ELP Camera Feed", frame)

        # Log frame count and FPS every 50 frames for faster feedback
        frame_count += 1
        if frame_count % 50 == 0:
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time
            print(f"Frame {frame_count}, Avg FPS: {avg_fps:.2f}")

        # Break on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    process.terminate()
    cv2.destroyAllWindows()
