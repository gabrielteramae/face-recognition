import cv2


class FaceDetector:
    def __init__(self, scale=1.1, min_neighbors=5, min_size=(60, 60)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise FileNotFoundError(cascade_path)
        self.scale = scale
        self.min_neighbors = min_neighbors
        self.min_size = tuple(min_size)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in rects]
