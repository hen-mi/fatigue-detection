import cv2
import mediapipe as mp
import time
import source

mpFaceMesh      =        mp.solutions.face_mesh
FaceMesh        =        mpFaceMesh.FaceMesh(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
mpDraw          =        mp.solutions.drawing_utils
drawingSpec     =        mpDraw.DrawingSpec(thickness=1, circle_radius=1)
camera          =        source.CameraSource()

camera.start()

if not camera.checkCamera():  # Check if the web cam has opened correctly
    print("failed to open cam")
else:
    print('cam opened on port {}'.format(camera.id))

    while camera.isOnline:
        frame = camera.frame()
        
        imgRGB  =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = FaceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
                                    
        if not camera.checkCamera():
            print('failed to capture frame on iter ')

        cv2.imshow('Camera', frame)

        camera.end()

    cv2.destroyAllWindows()




    
    