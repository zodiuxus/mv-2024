import cv2
import numpy as np

if __name__ == "__main__":
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        exit(0)
    
    while True:
        _, frame = video.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        thresh_low = np.array([0, 50, 60], dtype = np.uint8)
        thresh_high = np.array([30, 150, 255], dtype = np.uint8)

        mask = cv2.inRange(hsv, thresh_low, thresh_high)

        skin = cv2.bitwise_and(frame, frame, mask = mask)

        cv2.imshow("Filter", skin)
        cv2.imshow("Original", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()