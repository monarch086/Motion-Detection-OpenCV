# importing libraries
import cv2
import numpy
from datetime import datetime, timedelta

# Defining a function motionDetection
def motionDetection():
    # capturing video in real time
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    currentFileName = ''
    start_capture_time = datetime.now()
    writer = None
    MOTION_TIMEOUT_SEC = 10

    # reading frames sequentially
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():

        # difference between the frames
        diff = cv2.absdiff(frame1, frame2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            if currentFileName == '':
                start_capture_time = datetime.now()
                current_time = start_capture_time.strftime("%H-%M-%S")
                currentFileName = 'capture_' + current_time + '.avi'
                writer = cv2.VideoWriter(currentFileName, cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
        else:
            now = datetime.now()
            time_difference = now - start_capture_time

            if time_difference > timedelta(seconds = MOTION_TIMEOUT_SEC):
                currentFileName = ''
                if writer:
                    writer.release()

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 900:
                continue

            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "STATUS: {}".format('MOTION DETECTED'), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (217, 10, 10), 2)

        # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        if currentFileName != '':
            writer.write(frame1)

        cv2.imshow("Video", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(50) == 27: # Esc
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    motionDetection()