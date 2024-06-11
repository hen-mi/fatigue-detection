import cv2
'''
Handles frame by frame image capturing
'''

class CameraSource():
    def __init__(self, id=None):
        self.id         =        id
        self.isOnline   =     False
        self.capture    =      None
        
    def checkCamera(self):
        return self.capture is not None and self.capture.read()[0]

    def start(self):
        if not self.isOnline:
            if self.id is not None:
                self.capture = cv2.VideoCapture(self.id)
                if not self.checkCamera():
                    print(f'Camera {self.id} not working')
                    self.capture.release()
                    self.capture = None
                else:
                    self.isOnline = True
                    return
            else:
                for cameraid_ in range(0, 5000): #if no id is parsed, tries to find a camera
                    self.id= cameraid_
                    self.capture = cv2.VideoCapture(self.id)
                    
                    if self.checkCamera():
                        print(f'Camera {cameraid_} found and working')
                        self.isOnline = True
                        return
                
                # If no camera is found
                print('No working camera found')
                self.capture = None

    def frame(self):
        success, frame = self.capture.read()
        
        return frame
    
    def end(self):
        k = cv2.waitKey(1)
        if k == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            