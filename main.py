import time
import signal
import cv2
import threading
import argparse
from JetsonCamera import Camera
from Focuser import Focuser
from Autofocus import FocusState, doFocus
from models import FasterRCNN, Resnet
import torch
import numpy as np
from connector import StabConnector


exit_ = False
def sigint_handler(signum, frame):
    global exit_
    exit_ = True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Cuda is available: " + str(torch.cuda.is_available()))

    # Faster-Rcnn
    faster_rcnn = FasterRCNN('faster_rcnn_mobilenetv3_15.pth', device)
    faster_rcnn_threshold = 0.5
    faster_rcnn_color = (128, 128, 0)

    # Resnet
    resnet = Resnet('resnet18.pth', device)
    resnet_threshold = 0.5
    resnet_color = (0, 0, 255)
    resnet_area_width = 320
    resnet_area_height = 320

    # Setup connector
    stabConnector = StabConnector('192.168.2.223', port=0, data_block=1)
    print('Connected to PLC: {}'.format(str(stabConnector.is_available())))

    #width=1280, height=720
    display_width = 1280
    display_height = 720
    capture_width = 3840
    capture_height = 2160
    framerate = 21

    camera = Camera(display_width, display_height, capture_width, capture_height, 21)
    focuser = Focuser(7)

    display_width = 1280
    display_height = 720

    focusState = FocusState()
    doFocus(camera, focuser, focusState)

    prev_time = 0
    detected_defect = False

    while not exit_:

        if focusState.isFinish() and stabConnector.enableVisionCheck():
            try:
                frame = camera.getFrame(2000)
                image = faster_rcnn.preprocessing(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                predictions = faster_rcnn.predict(image.to(device))

                # Wyciągnij wyniki z predykcji
                boxes = predictions['boxes'].cpu().numpy()
                scores = predictions['scores'].cpu().numpy()

                # Wyodrębnij ramki ograniczające dla obiektów z wynikami powyżej progu
                filtered_indices = np.where(scores > faster_rcnn_threshold)[0]

                # image jest PIL po processingu 
                image = np.array(image)
   
                # Narysuj ramki ograniczające na klatce
                for box, score in zip(boxes[filtered_indices], scores[filtered_indices]):
                    x, y, x2, y2 = box
                    x, y, x2, y2 = int((x/resnet_area_width)*display_width), int((y/resnet_area_height)*display_height), int((x2/resnet_area_width)*display_width), int((y2/resnet_area_height)*display_height)

                    # score_text = "score: {}".format(str(round(score, 2)))
                    
                    item_image = frame[y:y2, x:x2]

                    if item_image.shape[1] < 128:
                        raise ValueError('To small object for resnet')

                    item_image = resnet.preprocessing(item_image)
                    item_image = item_image.unsqueeze(0)

                    resnet_results = resnet.predict(item_image.to(device), resnet_threshold)

                    if len(resnet_results) > 0:
                        detected_defect = True

                    cv2.putText(frame,
                                str(resnet_results), 
                                (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.4, 
                                resnet_color, 
                                1)
                    
                    cv2.rectangle(frame, (x, y), (x2, y2), faster_rcnn_color, 2)

                if detected_defect:
                    stabConnector.elementIsInvalid()

                else:
                    stabConnector.elementIsValid()

                detected_defect = False

                cv2.imwrite("last_test.png", frame)
                cv2.imshow("Faster RCNN", frame)

            except Exception as e:
                print('Error! - ' + str(e))
                camera.close()
                break

        frame = camera.getFrame(2000)

        # Wyświetl prędkość klatek
        curr_time = cv2.getTickCount()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            fps = cv2.getTickFrequency() / time_diff
            prev_time = curr_time
            cv2.putText(frame, "FPS: {:.0f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Stream", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            exit_ = True

        if key == ord('f'):
            if focusState.isFinish():
                focusState.reset()
                doFocus(camera, focuser, focusState)
                next_predict = True

            else:
                print("Focus is not done yet.")

    camera.close()
