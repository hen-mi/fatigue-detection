import cv2
from pathlib import Path
from settings import settings
'''
Performs the detection on videos in a given path
'''

class FrameSource: 
    
    def __init__(self, location: Path = settings.STATIC_FILE_PATH):
        self.location = location
        self.capture = None
        
    def start(self):
        self.capture = cv2.VideoCapture(str(self.location))
        
        if self.location is None or not self.capture.read()[0]:
            raise FileNotFoundError(f"Video File did not open at: {self.location}")

    def next_frame(self):
        return self.capture.read()
    
    def stop(self):
        self.capture.release()