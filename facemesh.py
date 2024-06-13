import cv2
import mediapipe as mp

class FaceMesh:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mpFaceMesh         =   mp.solutions.face_mesh
        self.faceMesh           =   self.mpFaceMesh.FaceMesh(min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence)
        self.mpDraw             =   mp.solutions.drawing_utils
        self.drawingSpec        =   self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    
    def process_frame(self, frame):
        
        imgRGB      =   cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results     =   self.faceMesh.process(imgRGB)
        
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(frame, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawingSpec, self.drawingSpec)
        return frame
