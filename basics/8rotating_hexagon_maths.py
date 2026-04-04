import cv2
import numpy as np
import math
import time

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Center of screen
    cx, cy = w // 2, h // 2
    radius = 100

    # Time-based rotation
    angle_offset = time.time() * 60   # speed of rotation

    # Store hexagon points
    points = []

    # Generate 6 points (hexagon)
    for i in range(6):
        angle = np.deg2rad(angle_offset + i * 60)  # 360/6 = 60°
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        points.append((x, y))

    # Draw lines between points
    for i in range(6):
        cv2.line(frame, points[i], points[(i+1) % 6], (0, 255, 255), 3)

    # Show
    cv2.imshow("Rotating Hexagon", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()