import cv2
import mediapipe as mp
import numpy as np
import fatiguedetector

RIGHT_EYE   = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
LEFT_EYE    = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

class FaceMesh:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mpFaceMesh         =   mp.solutions.face_mesh
        self.faceMesh           =   self.mpFaceMesh.FaceMesh(min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence)
        self.mpDraw             =   mp.solutions.drawing_utils
        self.drawingSpec        =   self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.fatiguedetector    =   fatiguedetector.FatigueDetector()
    
    def landmarksDetection(self, frame, results, draw=False):
        height, width, channel = frame.shape
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * width), int(point.y * height)) for point in results.multi_face_landmarks[0].landmark]
        if draw:
            for p in mesh_coord:
                cv2.circle(frame, p, 2, (255, 0, 0), -1)
        # returning the list of tuples for each landmarks 
        return mesh_coord     
    
    
    def processFrame(self, frame, flag):
        '''
        flag =  0 -> full face mesh
        flag =  1 -> eyes only mesh
        '''
        imgRGB      =   cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results     =   self.faceMesh.process(imgRGB)
        
        if results.multi_face_landmarks:
            #face
            if flag == 0:
                for faceLms in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(frame, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawingSpec, self.drawingSpec)
            
            #eyes only   
            elif flag == 1:
                mesh_coords = self.landmarksDetection(frame, results, False)
                cv2.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)

                left_eye_points = [mesh_coords[p] for p in fatiguedetector.LEFT]
                right_eye_points = [mesh_coords[p] for p in fatiguedetector.RIGHT]

                if left_eye_points and right_eye_points:
                    ear = self.fatiguedetector.average_ear(left_eye_points, right_eye_points)
                    is_fatigued = self.fatiguedetector.current_situation(ear)

                    if is_fatigued:
                        cv2.putText(frame, "FATIGUE DETECTED", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Draw eye polylines
                    cv2.polylines(frame, [np.array(left_eye_points, dtype=np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.polylines(frame, [np.array(right_eye_points, dtype=np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Display the blink count on the screen
                cv2.putText(frame, f"Blinks: {self.fatiguedetector.blink_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return frame
    
