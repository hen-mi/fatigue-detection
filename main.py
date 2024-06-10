import cv2
import mediapipe as mp
import time

mpFaceMesh      =        mp.solutions.face_mesh
FaceMesh        =        mpFaceMesh.FaceMesh(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
mpDraw          =        mp.solutions.drawing_utils
drawingSpec     =        mpDraw.DrawingSpec(thickness=1, circle_radius=1)
VideoCapture    =        cv2.VideoCapture(0)
i = 0

if not VideoCapture.isOpened():  # Check if the web cam has opened correctly
    print("failed to open cam")
else:
    print('cam opened on port {}'.format(0))
    print(i)
    while VideoCapture.isOpened():
        
        success, cv_frame = VideoCapture.read()
        imgRGB  =  cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        results = FaceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(cv_frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
                i = i+1
                print(i)
                    
        if not success:
            print(f'failed to capture frame on iter {i}', i)

        cv2.imshow('Camera', cv_frame)
        i = i+1
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    VideoCapture.release()
    cv2.destroyAllWindows()




    
    