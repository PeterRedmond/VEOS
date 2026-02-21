import serial
import time
import csv

# User configuration:
COM_PORT = 'COM8'           # Serial port for ESP32-C3 (adjust as needed for your system)
BAUD_RATE = 115200          # Serial baud rate (ESP32 USB-CDC ignores this value but we keep it)
TARGET_HZ = 100             # Base trigger frequency in Hz (100 Hz for this setup)
FRAME_COUNT_TARGET = 1000   # Number of frames to capture (i.e., number of trigger events to attempt)

# Calculate the period in seconds for the target frequency
PERIOD = 1.0 / TARGET_HZ    # 0.01 s for 100 Hz

# Open serial connection to the ESP32
ser = serial.Serial(COM_PORT, BAUD_RATE)
# Give the ESP32 a moment to reset (some boards reset on new serial connection)
time.sleep(2)  # 2 seconds wait for safety

# Optionally, send a startup sync pattern command to ESP32 (if using the 'S' pattern):
# ser.write(b'S')
# time.sleep(1)  # wait for pattern to finish (adjust delay if pattern is longer)

# Prepare CSV file for logging
csv_file = open("trigger_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
# Write CSV header
csv_writer.writerow(["Timestamp", "Frame", "TriggerCount", "CameraOK", "EEG", "EOG"])

# Initialize counters and timing variables
triggers_sent = 0          # count of triggers actually sent (frames captured)
next_trigger_time = time.perf_counter()  # next scheduled trigger time (start immediately)
tolerance = 0.001          # 1 ms tolerance to avoid skipping for tiny scheduling jitter

print("Starting trigger loop...")

# Main loop: run until we have sent the desired number of triggers
skip_count = 0
while triggers_sent < FRAME_COUNT_TARGET:
    now = time.perf_counter()
    # Wait until the scheduled trigger time
    if now < next_trigger_time:
        time.sleep(next_trigger_time - now)
        now = time.perf_counter()
    # Check if we missed the scheduled time by more than tolerance
    if now - next_trigger_time > tolerance:
        # We are behind schedule – skip this tick to catch up
        # Calculate how many intervals were missed
        missed_intervals = int((now - next_trigger_time) / PERIOD) + 1
        for i in range(missed_intervals):
            # Even if multiple intervals elapsed, mark each as skipped
            skipped_time = next_trigger_time  # the scheduled time for the skipped tick
            timestamp = skipped_time - next_trigger_time + (next_trigger_time - next_trigger_time)
            timestamp = skipped_time - (next_trigger_time - PERIOD)  # Correction for relative time calculation
            # Actually, simpler: compute timestamp relative to start of loop for log
            timestamp = skipped_time  # (We will record absolute monotonic time for simplicity)
            timestamp_rel = timestamp - 0.0  # We can treat the first trigger as time 0. Here we use absolute monotonic time.
            # Log the skip event (no frame, trigger count stays the same)
            csv_writer.writerow([f"{timestamp_rel:.6f}", "", f"{triggers_sent}", "0", "", ""])
            skip_count += 1
            # Advance the schedule to the next interval
            next_trigger_time += PERIOD
            # If we've skipped through the end of desired frames (unlikely), break
            # (We don't increment triggers_sent on skip, since no frame was captured)
        continue  # skip sending trigger this iteration
    # If on time (or within tolerance), send the trigger
    triggers_sent += 1
    frame_number = triggers_sent  # assign frame number (same as trigger count here)
    timestamp = next_trigger_time  # scheduled time for this trigger
    timestamp_rel = timestamp  # If we consider the first trigger at time 0, use relative time. (Monotonic zero reference)
    # Send trigger command to ESP32
    ser.write(b'T')
    # Simulate camera capture (replace this block with actual camera capture code):
    capture_duration = 0.003  # normal frame capture ~3 ms
    if triggers_sent % 10 == 0:
        # Simulate every 10th frame being slow (15 ms) to test skip logic
        capture_duration = 0.015
    time.sleep(capture_duration)
    # (If integrating a real camera, you would trigger the camera via the ESP32 and possibly grab a frame via an API here.
    # Ensure that grabbing the frame does not exceed the 10 ms window on average. If it does, the skip logic will handle it.)
    # Log the successful trigger event
    csv_writer.writerow([f"{timestamp_rel:.6f}", f"{frame_number}", f"{triggers_sent}", "1", "", ""])
    # Schedule next trigger time
    next_trigger_time += PERIOD

# Loop ended: all requested frames have been triggered (and any necessary skips logged)
print(f"Trigger loop completed. Sent {triggers_sent} triggers with {skip_count} skips.")

# Close serial and file resources
ser.close()
csv_file.close()
