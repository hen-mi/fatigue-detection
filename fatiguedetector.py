import facemesh
import cv2
import mediapipe as mp
import facemesh
import numpy as np
LEFT=  [33,160,158,133,153,144]
RIGHT = [362,385,387,263,373,380]
DROWSINESS_THRESHOLDS = [0.18,0.2,0.225,0.25]


class FatigueDetector:
    def __init__(self, thresholds=DROWSINESS_THRESHOLDS):
        self.thresholds = thresholds
        self.blink_count = 0
        self.current_threshold_index = 0
    
    def eye_aspect_ratio(self, eye_points):
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        ear = (A + B) / (2.0 * C)
        return ear
    
    def average_ear(self, left_eye_points, right_eye_points):
        left_ear = self.eye_aspect_ratio(left_eye_points)
        right_ear = self.eye_aspect_ratio(right_eye_points)
        return (left_ear + right_ear) / 2.0
    
    def current_situation(self, ear):
        if ear <= self.thresholds[self.current_threshold_index]:
            self.blink_count += 1
            self.current_threshold_index = min(self.current_threshold_index + 1, len(self.thresholds) - 1)
        else:
            self.current_threshold_index = 0

        return self.blink_count >= len(self.thresholds)