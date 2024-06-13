import cv2
import videosource
import argparse
import facemesh
import fatiguedetector
import numpy as np


camera=videosource.CameraSource()
camera.start()
fms = facemesh.FaceMesh()

if __name__ == "__main__":
    if not camera.checkCamera():  # Check if the webcam has opened correctly
        print("failed to open cam")
    else:
        print('cam opened on port {}'.format(camera.id))

        while camera.isOnline:
            frame = camera.frame()
            processedframe = fms.processFrame(frame, 1)

            if not camera.checkCamera():
                print('failed to capture frame on iter')

            cv2.imshow('Camera', frame)
            if camera.end():
                break

    camera.stop()
    cv2.destroyAllWindows()