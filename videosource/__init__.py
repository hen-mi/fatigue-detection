from typing import Protocol
from .camera import CameraSource

class Source(Protocol):
    def start(self):
        ...
    
    def end(self):
        ...
        
    def frame(self):
        ...
        
    def checkCamera(self):
        ...