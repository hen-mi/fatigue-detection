import cv2
import videosource
import argparse
import facemesh


camera=videosource.CameraSource()
camera.start()
fms = facemesh.faceMesh()

if __name__ == "__main__":

    if not camera.checkCamera():  # Check if the web cam has opened correctly
        print("failed to open cam")
    else:
        print('cam opened on port {}'.format(camera.id))

        while camera.isOnline:
            frame = camera.frame()
            processedframe = fms.processFrame(frame, 1)
                                        
            if not camera.checkCamera():
                print('failed to capture frame on iter ')

            cv2.imshow('Camera', processedframe)
            if camera.end():
                break

    
    