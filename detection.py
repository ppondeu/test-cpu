import cv2
import datetime
import numpy as np

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
red_bgr = (0, 0, 255)
blue_bgr = (255, 0, 0)
green_bgr = (0, 255, 0)
yellow_bgr = (0, 255, 255)
cv2.namedWindow("Application", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Application", 1080, 720)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def passing_percentage(opp_count, team_count):
    if opp_count == 0 and team_count == 0:
        return f"Free space", green_bgr
    elif opp_count == 0 and team_count > 0:
        return f"Free to pass", green_bgr
    elif opp_count > team_count:
        return f"Beware opponent is near", red_bgr
    elif opp_count < team_count:
        return f"Depends on your decision", yellow_bgr
    else:
        return f"Depends on your decision", yellow_bgr

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    # if hue >= 165:  # Upper limit for divided red hue
    #     lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    #     upperLimit = np.array([180, 255, 255], dtype=np.uint8)
        
    # elif hue <= 15:  # Lower limit for divided red hue
    #     lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
    #     upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    # else:
    #     lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    #     upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    # Handle blue hue wrap-around
    if 165 <= hue <= 180:  # Upper limit for divided blue hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif 0 <= hue <= 15:  # Lower limit for divided blue hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

def main():
    cap = cv2.VideoCapture(0) 
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        opp_count = 0
        team_count = 0

        for i in np.arange(0, person_detections.shape[2]):
            # print(person_detections.shape)
            confidence = person_detections[0, 0, i, 2]
         
            if confidence > 0.8:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue
                else:
                    person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box.astype("int")
                    
                    result = frame[startY:endY, startX:endX]

                    if result.any():
                        hsvImage = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

                        lowerLimit, upperLimit = get_limits(color=blue_bgr)

                        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
                        interested = cv2.bitwise_and(result,result,mask=mask)

                        non_black_pixels = np.sum(np.any(interested != [0, 0, 0], axis=1))
                        if non_black_pixels > 250:
                            cv2.rectangle(frame, (startX, startY), (endX, endY), blue_bgr, 2)
                            # print(f"Detecting Red {non_black_pixels}")
                            team_count += 1
                            cv2.putText(frame, f"Ally : {team_count}", (startX, startY - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, blue_bgr, 1)
                        else:
                            cv2.rectangle(frame, (startX, startY), (endX, endY), green_bgr, 2)
                            # print(f"Not Detecting Red {non_black_pixels}")
                            opp_count += 1
                            cv2.putText(frame, f"Opp : {opp_count}", (startX, startY - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, green_bgr, 1)
                            
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, red_bgr, 1)
        txt, color_bgr = passing_percentage(opp_count, team_count)
        cv2.putText(frame, txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_bgr, 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        print(txt)
        print(f"red : {opp_count}")
        print(f"blue : {team_count}")

    cap.release()
    cv2.destroyAllWindows()

main()
