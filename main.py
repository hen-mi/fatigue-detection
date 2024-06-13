import cv2
import videosource
import argparse
import facemesh


camera=videosource.CameraSource()
camera.start()
face_mesh_detector = facemesh.FaceMesh()


if not camera.checkCamera():  # Check if the web cam has opened correctly
    print("failed to open cam")
else:
    print('cam opened on port {}'.format(camera.id))

    while camera.isOnline:
        frame = camera.frame()
        processed_frame = face_mesh_detector.process_frame(frame)
                                    
        if not camera.checkCamera():
            print('failed to capture frame on iter ')

        cv2.imshow('Camera', processed_frame)
        if camera.end():
            break





    
    